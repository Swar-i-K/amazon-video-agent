"""Thin wrapper around the Groq API (key comes from GROQ_API_KEY)."""
from __future__ import annotations

import base64
import os
import time

import cv2
import numpy as np


def _client():
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is not set (add it as a GitHub secret).")
    import groq
    return groq.Groq()


def jpeg_b64(img_bgr: np.ndarray, max_side: int = 1280) -> str:
    h, w = img_bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def ask(model: str, system: str, user: str, images: list[np.ndarray] | None = None,
        max_tokens: int = 8000, temperature: float = 0.8) -> str:
    import groq
    client = _client()

    if images:
        content: list[dict] | str = [{"type": "text", "text": user}]
        for im in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64(im)}"},
            })
    else:
        content = user  # plain string for text-only calls

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": content}]

    last = None
    for attempt in range(4):
                try:
            resp = client.chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=temperature,
                messages=messages,
            )
            choice = resp.choices[0]
            text = choice.message.content or ""
            # drop any <think>...</think> reasoning that leaks into the reply
            import re
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            if choice.finish_reason == "length":
                print(f"[llm] WARNING: {model} hit max_tokens={max_tokens}; reply was cut off. "
                      f"Raw start: {text[:300]!r}", flush=True)
            return text
    raise RuntimeError(f"Anthropic API failed: {last}")
