"""Text overlays (captions, callouts, disclosure banner, end card) drawn with Pillow, blended per frame."""
from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .common import log
from .templates import tpl_of


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
               accent=None, align="center", line_gap=8, stroke=(1, (0, 0, 0, 200))):
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
        d.text((x, y), l, font=font, fill=fg + (255,), stroke_width=stroke[0], stroke_fill=stroke[1])
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


# ----------------------------- template styling -----------------------------
class GlassOverlay:
    """A frosted-glass panel: blurs whatever is behind it, then adds a light tint and a thin border."""

    def __init__(self, start, end, x, y, w, h, radius=22, fade=0.25, blur=9.0):
        self.start, self.end, self.fade, self.blur = start, end, fade, blur
        self.x, self.y, self.w, self.h = x, y, w, h
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
        self.mask = (np.asarray(m, np.float32) / 255.0)[..., None]
        ring = Image.new("L", (w, h), 0)
        ImageDraw.Draw(ring).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, outline=255, width=2)
        self.ring = (np.asarray(ring, np.float32) / 255.0)[..., None]

    def apply(self, frame: np.ndarray, t: float) -> None:
        if not (self.start <= t <= self.end):
            return
        m = max(0.0, min(1.0, (t - self.start) / self.fade, (self.end - t) / self.fade))
        reg = frame[self.y:self.y + self.h, self.x:self.x + self.w]
        if reg.shape[0] != self.h or reg.shape[1] != self.w:
            return
        blurred = cv2.GaussianBlur(reg, (0, 0), self.blur).astype(np.float32)
        panel = blurred * 0.60 + 255.0 * 0.12
        a = self.mask * m
        out = reg.astype(np.float32) * (1 - a) + panel * a
        out = out + self.ring * m * 70.0
        reg[:] = np.clip(out, 0, 255).astype(np.uint8)


def _lum(rgb) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def ink_for(rgb) -> tuple:
    return (20, 20, 24) if _lum(rgb) > 150 else (255, 255, 255)


def panel_look(tpl: dict) -> dict:
    """Box/ink used for chips, callouts and banners so they match the template (light templates get light panels)."""
    if tpl["caption"]["style"] == "light":
        return {"box": (255, 255, 255, 236), "fg": (24, 26, 31), "stroke": (0, (0, 0, 0, 0))}
    return {"box": (10, 10, 12, 205), "fg": (255, 255, 255), "stroke": (1, (0, 0, 0, 200))}


def caption_overlays(chunk, font, W, H, tpl, t0, t1) -> list:
    style, acc = tpl["caption"]["style"], tpl["accent"]
    max_w = int(W * 0.78)
    y_base = int(H * (0.885 if style == "outline" else 0.86))
    span_end, fade = t1, 0.08
    if style == "glass":
        txt = text_patch(chunk, font, max_w, box=(0, 0, 0, 0), stroke=(1, (0, 0, 0, 120)), pad=(32, 18))
        x, y = (W - txt.width) // 2, y_base - txt.height
        return [GlassOverlay(t0, span_end, x, y, txt.width, txt.height, radius=24, fade=fade),
                _to_overlay(txt, x, y, t0, span_end, fade=fade)]
    if style == "light":
        img = text_patch(chunk, font, max_w, fg=(24, 26, 31), box=(255, 255, 255, 230), radius=16, stroke=(0, (0, 0, 0, 0)))
    elif style == "solid":
        img = text_patch(chunk, font, max_w, fg=ink_for(acc), box=acc + (240,), radius=12, stroke=(0, (0, 0, 0, 0)))
    elif style == "outline":
        img = text_patch(chunk, font, max_w, box=(0, 0, 0, 0), pad=(14, 8), stroke=(max(2, int(font.size * 0.07)), (0, 0, 0, 235)))
    else:  # pill
        img = text_patch(chunk, font, max_w, box=(0, 0, 0, 150))
    return [_to_overlay(img, (W - img.width) // 2, y_base - img.height, t0, span_end, fade=fade)]


def badge_overlays(num_text, sub, tpl, W, H, t0, t1, fonts) -> list:
    """The big '#3' rank reveal at the start of each item in a list video."""
    big, small = fonts
    acc, style = tpl["accent"], tpl["badge"]
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    sub_lines = _wrap(tmp, sub, small, int(W * 0.30)) if sub else []
    nw = int(tmp.textlength(num_text, font=big))
    asc, desc = big.getmetrics()
    sl = small.getmetrics()
    sub_h = (sl[0] + sl[1] + 6) * len(sub_lines)
    pad = 28
    w = max(nw, max((int(tmp.textlength(l, font=small)) for l in sub_lines), default=0)) + 2 * pad
    h = asc + desc + sub_h + 2 * pad - (10 if sub_lines else 0)
    x, y = int(W * 0.05), int(H * 0.30)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    num_col, sub_col = (255, 255, 255), (235, 235, 235)
    glass = False
    if style == "solid":
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=26, fill=acc + (242,))
        num_col = sub_col = ink_for(acc)
    elif style == "outline":
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=26, fill=(255, 255, 255, 238), outline=acc + (255,), width=6)
        num_col, sub_col = acc, (30, 32, 38)
    elif style == "glow":
        glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(glow).text(((w - nw) // 2, pad - 6), num_text, font=big, fill=acc + (255,))
        glow = glow.filter(ImageFilter.GaussianBlur(14))
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=26, fill=(8, 9, 12, 220), outline=acc + (255,), width=3)
        img = Image.alpha_composite(img, glow)
        d = ImageDraw.Draw(img)
        num_col = acc
    else:  # glass
        glass = True
    d.text(((w - nw) // 2, pad - 6), num_text, font=big, fill=num_col + (255,), stroke_width=2 if glass else 0,
           stroke_fill=(0, 0, 0, 120))
    yy = pad - 6 + asc + desc
    for l in sub_lines:
        d.text(((w - d.textlength(l, font=small)) / 2, yy), l, font=small, fill=sub_col + (255,),
               stroke_width=1 if glass else 0, stroke_fill=(0, 0, 0, 120))
        yy += sl[0] + sl[1] + 6
    out = []
    if glass:
        out.append(GlassOverlay(t0, t1, x, y, w, h, radius=26, fade=0.4))
    out.append(_to_overlay(img, x, y, t0, t1, fade=0.4))
    return out


# ----------------------------- the overlay set ------------------------------
def build_overlays(rows, tl_total, cfg, W, H, product_title, extra_lines, ai_spans=None, product_labels=None,
                   endcard_line="Link in the description", kind=None, rank_notes=None):
    """kind: None (single product), 'compare' or 'list'. rank_notes: per-product subtitle for the list rank badge."""
    tpl = tpl_of(cfg)
    acc = tpl["accent"]
    look = panel_look(tpl)
    cap_font = load_font(cfg, int(H * (0.052 if tpl["caption"]["style"] == "solid" else 0.048)))
    call_font = load_font(cfg, int(H * 0.058))
    small = load_font(cfg, int(H * 0.030))
    big = load_font(cfg, int(H * 0.070))
    mid = load_font(cfg, int(H * 0.040))
    d = cfg["disclosure"]
    mode = d.get("mode", "full")
    ovs: list = []
    for r in rows:
        s = r.seg
        t0, t1 = r.start, r.start + r.dur
        if not s.get("endcard"):
            chunks = split_caption(s["text"])
            tot = sum(len(c) for c in chunks)
            t = t0
            for c in chunks:
                span = r.dur * len(c) / tot
                ovs += caption_overlays(c, cap_font, W, H, tpl, t, t + span + 0.02)
                t += span
        if s.get("onscreen"):
            img = text_patch(s["onscreen"], call_font, int(W * 0.5), fg=look["fg"], box=look["box"], accent=acc, align="left",
                             stroke=look["stroke"])
            ovs.append(_to_overlay(img, int(W * 0.05), int(H * 0.14), t0 + 0.4, min(t1, t0 + 4.8), fade=0.3))
        if s.get("banner"):
            img = text_patch(d["on_screen"], mid, int(W * 0.82), fg=look["fg"], box=look["box"], accent=acc, align="left",
                             stroke=look["stroke"])
            ovs.append(_to_overlay(img, (W - img.width) // 2, int(H * 0.05), t0 - 0.1, t1 + 0.4, fade=0.3))
        if s.get("endcard"):
            ovs.append(_endcard(s_start=t0 + 0.3, end=tl_total, W=W, H=H, title=product_title,
                                lines=[endcard_line] + ([d["on_screen"]] if mode != "description_only" else []) + extra_lines,
                                big=big, mid=mid, small=small, tpl=tpl))
    if product_labels:  # comparison / list videos: a chip while a line is about one product
        spans: list[list] = []
        for r in rows:
            pr = str(r.seg.get("product", "all"))
            if pr.isdigit() and 1 <= int(pr) <= len(product_labels):
                a, b = r.start, r.start + r.dur + r.pad * 0.6
                if spans and spans[-1][2] == pr and a - spans[-1][1] < 1.2:
                    spans[-1][1] = b
                else:
                    spans.append([a, b, pr])
        chip_font = load_font(cfg, int(H * 0.036))
        badge_fonts = (load_font(cfg, int(H * 0.17)), load_font(cfg, int(H * 0.034)))
        for a, b, pr in spans:
            i = int(pr)
            tag = f"#{i}" if kind == "list" else "ABC"[i - 1]
            chip = text_patch(f"{tag}  |  {product_labels[i - 1]}", chip_font, int(W * 0.5), fg=look["fg"], box=look["box"],
                              accent=acc, align="left", pad=(22, 12), radius=18, stroke=look["stroke"])
            ovs.append(_to_overlay(chip, int(W * 0.02), int(H * 0.03), a, b, fade=0.3))
            if kind == "list":      # big rank reveal for the first few seconds of each item
                sub = (rank_notes[i - 1] if rank_notes and rank_notes[i - 1] else product_labels[i - 1])
                ovs += badge_overlays(f"#{i}", sub, tpl, W, H, a + 0.15, min(b, a + 3.4), badge_fonts)
    if ai_spans:  # merge touching spans so the tag does not flicker between neighbouring AI shots
        merged: list[list[float]] = []
        for a, b in sorted(ai_spans):
            if merged and a <= merged[-1][1] + 0.5:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        ai_tag = text_patch(d.get("ai_tag", "AI-generated scene"), small, int(W * 0.3), fg=look["fg"], box=look["box"],
                            pad=(18, 8), radius=14, stroke=look["stroke"])
        for a, b in merged:
            ovs.append(_to_overlay(ai_tag, int(W * 0.02), int(H * 0.925) - ai_tag.height // 2, a + 0.2, b, fade=0.3))
    if mode == "on_screen":   # silent text near the start (no voice-over line)
        img = text_patch(d["on_screen"], mid, int(W * 0.82), fg=look["fg"], box=look["box"], accent=acc, align="left",
                         stroke=look["stroke"])
        ovs.append(_to_overlay(img, (W - img.width) // 2, int(H * 0.05), 1.0, min(7.5, tl_total - 1), fade=0.3))
    if mode in ("full", "on_screen"):
        tag = text_patch(d["tag"], small, int(W * 0.3), fg=look["fg"], box=look["box"], pad=(18, 8), radius=14, stroke=look["stroke"])
        ovs.append(_to_overlay(tag, W - tag.width - int(W * 0.02), int(H * 0.025), 0.6, 22.0, fade=0.5))
    return ovs


def _endcard(s_start, end, W, H, title, lines, big, mid, small, tpl=None):
    bg = tuple((tpl or {}).get("endcard", {}).get("bg", (8, 8, 10, 175)))
    ink = tuple((tpl or {}).get("endcard", {}).get("text", (255, 255, 255)))
    acc = (tpl or {}).get("accent", (255, 214, 120))
    first = acc if _lum(bg[:3]) < 150 else tuple(int(c * 0.8) for c in acc)
    img = Image.new("RGBA", (W, H), bg)
    d = ImageDraw.Draw(img)
    y = int(H * 0.28)
    for l in _wrap(d, title, big, int(W * 0.8))[:2]:
        d.text(((W - d.textlength(l, font=big)) / 2, y), l, font=big, fill=ink + (255,))
        y += int(H * 0.09)
    y += int(H * 0.04)
    for i, l in enumerate(lines):
        f = mid if i == 0 else small
        d.text(((W - d.textlength(l, font=f)) / 2, y), l, font=f, fill=(first if i == 0 else ink) + (255,))
        y += int(H * (0.065 if i == 0 else 0.045))
    return _to_overlay(img, 0, 0, s_start, end + 1, fade=0.7)
