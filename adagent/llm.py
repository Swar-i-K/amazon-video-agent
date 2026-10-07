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
            return resp.choices[0].message.content or ""
        except groq.AuthenticationError as e:  # bad key: retrying won't help
            raise SystemExit(f"Groq rejected GROQ_API_KEY: {e}")
        except Exception as e:  # network / rate limit / transient
            last = e
            time.sleep(2 + attempt * 8)  # longer backoff: Groq free tier has per-minute caps
    raise RuntimeError(f"Groq API failed: {last}")
    raise RuntimeError(f"Anthropic API failed: {last}")
