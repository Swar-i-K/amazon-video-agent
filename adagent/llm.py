"""LLM wrapper. Provider is chosen in config.yaml (llm.provider): "groq" (default) or "anthropic".

Keys come from environment variables:  GROQ_API_KEY  /  ANTHROPIC_API_KEY
List the Groq models your key can use:   python -m adagent.llm
"""
from __future__ import annotations

import base64
import os
import time
from functools import lru_cache

import cv2
import numpy as np
import requests

from .common import load_config, log

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

PREF_TEXT = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3-32b", "llama-3.1-8b-instant"]
PREF_VISION = ["meta-llama/llama-4-scout-17b-16e-instruct", "meta-llama/llama-4-maverick-17b-128e-instruct"]
_NOT_TEXT = ("whisper", "tts", "guard", "orpheus", "playai", "embed", "safeguard", "vision", "scout", "maverick", "-vl", "compound")
_substitutes: dict = {}   # (requested model, is_vision) -> model that actually works on this account


@lru_cache(maxsize=1)
def _llm_cfg() -> dict:
    return load_config().get("llm", {})


def jpeg_b64(img_bgr: np.ndarray, max_side: int = 1280) -> str:
    h, w = img_bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


# ------------------------------- Groq model discovery -----------------------
def groq_models(key: str) -> list[str]:
    r = requests.get(GROQ_MODELS_URL, headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    return sorted(m["id"] for m in r.json().get("data", []))


def _pick_substitute(model: str, vision: bool, ids: list[str]) -> str | None:
    if model in ids:
        return model
    for m in (PREF_VISION if vision else PREF_TEXT):
        if m in ids:
            return m
    if vision:
        cands = [i for i in ids if any(t in i.lower() for t in ("scout", "maverick", "vision", "-vl"))]
    else:
        cands = [i for i in ids if not any(t in i.lower() for t in _NOT_TEXT)]
    return cands[0] if cands else None


def _model_missing(status: int, text: str) -> bool:
    t = text.lower()
    return status in (400, 403, 404) and any(k in t for k in ("model_not_found", "does not exist", "decommissioned", "no access", "not found"))


# ------------------------------- Groq ---------------------------------------
_REASON_KEYS = ("reasoning_effort", "reasoning_format", "include_reasoning")


def _apply_reasoning(body: dict) -> None:
    """Reasoning models spend the token budget 'thinking' and can cut the JSON off. Keep the thinking short/hidden."""
    for k in _REASON_KEYS:
        body.pop(k, None)
    m = body["model"].lower()
    if "gpt-oss" in m:
        body.update(reasoning_effort="low", include_reasoning=False)
    elif any(t in m for t in ("qwen3", "qwq", "deepseek-r1")):
        body.update(reasoning_format="hidden")


def _ask_groq(model, system, user, images, max_tokens, temperature) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY is not set (add it as a GitHub secret).")
    max_tokens = min(max_tokens, int(_llm_cfg().get("groq_max_tokens", 6500)))  # free-tier limits count requested tokens
    vision = bool(images)
    if vision:  # instructions go in the user turn: most reliable with image input
        content = [{"type": "text", "text": system + "\n\n" + user}]
        for im in images:
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + jpeg_b64(im, 1024)}})
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    requested = model
    body = {"model": _substitutes.get((requested, vision), requested), "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature, "response_format": {"type": "json_object"}}
    _apply_reasoning(body)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last, swapped = "", False
    for attempt in range(8):
        try:
            r = requests.post(GROQ_URL, headers=headers, json=body, timeout=180)
        except requests.RequestException as e:
            last = str(e)
            time.sleep(3 + attempt * 4)
            continue
        if r.status_code == 200:
            choice = r.json()["choices"][0]
            if choice.get("finish_reason") == "length":
                log("  WARNING: the model hit its token limit and the reply was cut off.")
            return choice["message"].get("content") or ""
        last = f"{r.status_code}: {r.text[:300]}"
        if _model_missing(r.status_code, r.text) and not swapped:
            swapped = True
            ids = groq_models(key)
            log(f"  Groq model '{body['model']}' is not available to your account. Your models: {', '.join(ids)}")
            sub = _pick_substitute(requested, vision, ids)
            if not sub or sub == body["model"]:
                raise RuntimeError(f"No usable {'vision' if vision else 'text'} model on your Groq account. "
                                   f"Set llm.model / llm.vision_model in config.yaml to one of: {', '.join(ids)}")
            log(f"  Using '{sub}' instead (set it in config.yaml to make this permanent).")
            _substitutes[(requested, vision)] = sub
            body["model"] = sub
            _apply_reasoning(body)
            continue
        if r.status_code == 400 and any(k in body for k in _REASON_KEYS) and "reason" in r.text.lower():
            for k in _REASON_KEYS:  # this model does not accept the reasoning options
                body.pop(k, None)
            continue
        if r.status_code == 400 and "response_format" in body and "json" in r.text.lower():
            body.pop("response_format")  # model without JSON mode: our parser copes with plain text
            continue
        if r.status_code == 429:
            wait = min(65.0, float(r.headers.get("retry-after", 8 + attempt * 6)))
            log(f"  Groq rate limit, waiting {wait:.0f}s ...")
            time.sleep(wait)
            continue
        if r.status_code == 413:
            raise RuntimeError("Groq says this request is too large for your plan's per-minute token limit. "
                               "Lower groq_max_tokens in config.yaml, or retry in a minute. " + last)
        if r.status_code >= 500:
            time.sleep(3 + attempt * 4)
            continue
        raise RuntimeError(f"Groq error {last}")
    raise RuntimeError(f"Groq kept failing: {last}")


# ------------------------------- Anthropic (optional) -----------------------
def _ask_anthropic(model, system, user, images, max_tokens, temperature) -> str:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set (add it as a GitHub secret).")
    import anthropic
    client = anthropic.Anthropic()
    content: list[dict] = []
    for im in images or []:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg_b64(im)}})
    content.append({"type": "text", "text": user})
    last = None
    for attempt in range(4):
        try:
            resp = client.messages.create(model=model, max_tokens=max_tokens, temperature=temperature,
                                          system=system, messages=[{"role": "user", "content": content}])
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        except Exception as e:
            last = e
            time.sleep(2 + attempt * 4)
    raise RuntimeError(f"Anthropic API failed: {last}")


def ask(model: str, system: str, user: str, images: list[np.ndarray] | None = None,
        max_tokens: int = 8000, temperature: float = 0.8) -> str:
    fn = _ask_groq if _llm_cfg().get("provider", "groq") == "groq" else _ask_anthropic
    return fn(model, system, user, images, max_tokens, temperature)


if __name__ == "__main__":
    k = os.environ.get("GROQ_API_KEY")
    if not k:
        raise SystemExit("Set GROQ_API_KEY first.")
    print("\n".join(groq_models(k)))
