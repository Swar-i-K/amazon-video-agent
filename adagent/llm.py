"""Thin wrapper around the Anthropic API (key comes from ANTHROPIC_API_KEY)."""
from __future__ import annotations

import base64
import os
import time

import cv2
import numpy as np


def _client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set (add it as a GitHub secret).")
    import anthropic
    return anthropic.Anthropic()


def jpeg_b64(img_bgr: np.ndarray, max_side: int = 1280) -> str:
    h, w = img_bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def ask(model: str, system: str, user: str, images: list[np.ndarray] | None = None,
        max_tokens: int = 8000, temperature: float = 0.8) -> str:
    client = _client()
    content: list[dict] = []
    for im in images or []:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg_b64(im)}})
    content.append({"type": "text", "text": user})
    last = None
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=model, max_tokens=max_tokens, temperature=temperature,
                system=system, messages=[{"role": "user", "content": content}],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        except Exception as e:  # network/rate limit
            last = e
            time.sleep(2 + attempt * 4)
    raise RuntimeError(f"Anthropic API failed: {last}")
