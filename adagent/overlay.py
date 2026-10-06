"""Text overlays (captions, callouts, disclosure banner, end card) drawn with Pillow, blended per frame."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import log


def load_font(cfg: dict, size: int):
    from pathlib import Path
    for f in cfg["video"]["fonts"]:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    log("  (no TTF font found, using Pillow default - install fonts-dejavu-core)")
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


@dataclass
class Overlay:
    start: float
    end: float
    x: int
    y: int
    prem: np.ndarray    # premultiplied BGR float32 (h,w,3)
    inv: np.ndarray     # 1 - alpha float32 (h,w,1)
    fade: float = 0.25

    def apply(self, frame: np.ndarray, t: float) -> None:
        if not (self.start <= t <= self.end):
            return
        m = min(1.0, (t - self.start) / self.fade, (self.end - t) / self.fade)
        m = max(0.0, m)
        h, w = self.prem.shape[:2]
        reg = frame[self.y:self.y + h, self.x:self.x + w]
        out = reg.astype(np.float32) * (1 - (1 - self.inv) * m) + self.prem * m
        reg[:] = np.clip(out, 0, 255).astype(np.uint8)


def _wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def text_patch(text, font, max_w, fg=(255, 255, 255), box=(0, 0, 0, 165), pad=(30, 18), radius=24,
               accent=None, align="center", line_gap=8):
    """Return RGBA PIL image containing wrapped text on a rounded box."""
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    lines = _wrap(tmp, text, font, max_w - 2 * pad[0])
    asc, desc = font.getmetrics()
    lh = asc + desc
    tw = max(int(tmp.textlength(l, font=font)) for l in lines)
    w = tw + 2 * pad[0] + (14 if accent else 0)
    h = len(lines) * lh + (len(lines) - 1) * line_gap + 2 * pad[1]
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=box)
    if accent:
        d.rounded_rectangle([0, 0, 12, h - 1], radius=6, fill=accent + (255,))
    x0 = pad[0] + (14 if accent else 0)
    y = pad[1]
    for l in lines:
        lw = d.textlength(l, font=font)
        x = x0 + ((tw - lw) / 2 if align == "center" else 0)
        d.text((x, y), l, font=font, fill=fg + (255,), stroke_width=1, stroke_fill=(0, 0, 0, 200))
        y += lh + line_gap
    return img


def _to_overlay(img: Image.Image, x: int, y: int, start: float, end: float, fade: float = 0.25) -> Overlay:
    a = np.asarray(img, dtype=np.float32)
    alpha = a[..., 3:4] / 255.0
    bgr = a[..., [2, 1, 0]]
    return Overlay(start, end, x, y, bgr * alpha, 1.0 - alpha, fade)


def _split_words(w: list[str], max_words: int) -> list[list[str]]:
    """Recursively split a word list near its middle, preferring a comma/conjunction boundary."""
    if len(w) <= max_words:
        return [w]
    mid = len(w) / 2
    cands = [i for i in range(3, len(w) - 2) if w[i - 1].endswith((",", ";", ":", "-"))]
    cands += [i for i in range(3, len(w) - 2) if w[i].lower() in ("and", "but", "so", "because", "which", "or")]
    i = min(cands, key=lambda c: abs(c - mid)) if cands else int(round(mid))
    return _split_words(w[:i], max_words) + _split_words(w[i:], max_words)


def split_caption(text: str, max_words: int = 9) -> list[str]:
    parts = re.split(r"(?<=[.!?;:])\s+", text.strip())
    chunks: list[str] = []
    for p in parts:
        for piece in _split_words(p.split(), max_words):
            if piece:
                chunks.append(" ".join(piece))
    return chunks


def build_overlays(rows, tl_total, cfg, W, H, product_title, extra_lines):
    acc = tuple(cfg["video"]["accent_color"])
    cap_font = load_font(cfg, int(H * 0.048))
    call_font = load_font(cfg, int(H * 0.058))
    small = load_font(cfg, int(H * 0.030))
    big = load_font(cfg, int(H * 0.070))
    mid = load_font(cfg, int(H * 0.040))
    d = cfg["disclosure"]
    ovs: list[Overlay] = []
    for r in rows:
        s = r.seg
        t0, t1 = r.start, r.start + r.dur
        if not s.get("endcard"):
            chunks = split_caption(s["text"])
            tot = sum(len(c) for c in chunks)
            t = t0
            for c in chunks:
                span = r.dur * len(c) / tot
                img = text_patch(c, cap_font, int(W * 0.78), box=(0, 0, 0, 150))
                ovs.append(_to_overlay(img, (W - img.width) // 2, int(H * 0.86) - img.height, t, t + span + 0.02, fade=0.08))
                t += span
        if s.get("onscreen"):
            img = text_patch(s["onscreen"], call_font, int(W * 0.5), box=(10, 10, 10, 190), accent=acc, align="left")
            ovs.append(_to_overlay(img, int(W * 0.05), int(H * 0.14), t0 + 0.4, min(t1, t0 + 4.8), fade=0.3))
        if s.get("banner"):
            img = text_patch(d["on_screen"], mid, int(W * 0.82), box=(20, 20, 20, 215), accent=acc, align="left")
            ovs.append(_to_overlay(img, (W - img.width) // 2, int(H * 0.05), t0 - 0.1, t1 + 0.4, fade=0.3))
        if s.get("endcard"):
            ovs.append(_endcard(s_start=t0 + 0.3, end=tl_total, W=W, H=H, title=product_title,
                                lines=["Link in the description", d["on_screen"]] + extra_lines,
                                big=big, mid=mid, small=small))
    tag = text_patch(d["tag"], small, int(W * 0.3), box=(0, 0, 0, 140), pad=(18, 8), radius=14)
    ovs.append(_to_overlay(tag, W - tag.width - int(W * 0.02), int(H * 0.025), 0.6, 22.0, fade=0.5))
    return ovs


def _endcard(s_start, end, W, H, title, lines, big, mid, small):
    img = Image.new("RGBA", (W, H), (8, 8, 10, 175))
    d = ImageDraw.Draw(img)
    y = int(H * 0.28)
    for l in _wrap(d, title, big, int(W * 0.8))[:2]:
        d.text(((W - d.textlength(l, font=big)) / 2, y), l, font=big, fill=(255, 255, 255, 255))
        y += int(H * 0.09)
    y += int(H * 0.04)
    for i, l in enumerate(lines):
        f = mid if i == 0 else small
        d.text(((W - d.textlength(l, font=f)) / 2, y), l, font=f, fill=(255, 214, 120, 255) if i == 0 else (235, 235, 235, 255))
        y += int(H * (0.065 if i == 0 else 0.045))
    return _to_overlay(img, 0, 0, s_start, end + 1, fade=0.7)
