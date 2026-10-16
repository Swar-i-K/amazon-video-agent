"""Visual templates: one, two, three, four, best.

A template sets the caption style, accent colour, colour grade, letterbox bars, film grain, shot transitions and the
backdrop look used by the cinematic product shots. Templates are defined in config.yaml (templates:), so you can tweak them.
"""
from __future__ import annotations

import copy

import cv2
import numpy as np

DEFAULT_KEY = "best"


def smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def template_keys(cfg: dict) -> list[str]:
    return list((cfg.get("templates") or {}).keys())


def resolve_template(cfg: dict, key: str | None = None) -> dict:
    tpls = cfg.get("templates") or {}
    key = key or cfg["video"].get("template") or DEFAULT_KEY
    if key not in tpls:
        raise SystemExit(f"Unknown template '{key}'. Options: {', '.join(tpls)}")
    t = copy.deepcopy(tpls[key])
    t["key"] = key
    t["accent"] = tuple(int(x) for x in t["accent"])
    t.setdefault("caption", {"style": "pill"})
    t.setdefault("grade", {})
    t.setdefault("transition", {"kind": "fade", "seconds": 0.35})
    t.setdefault("backdrop", "studio")
    t.setdefault("badge", "solid")
    t.setdefault("endcard", {"bg": [8, 8, 10, 175], "text": [255, 255, 255]})
    return t


def apply_template(cfg: dict, key: str | None = None) -> dict:
    """Select the template for this run (also sets the shot cross-fade length so shot planning matches)."""
    t = resolve_template(cfg, key)
    cfg["_tpl"] = t
    cfg["video"]["crossfade"] = float(t["transition"].get("seconds", 0.35))
    return t


def tpl_of(cfg: dict) -> dict:
    t = cfg.get("_tpl")
    return t if t is not None else resolve_template(cfg)


# ----------------------------- colour grade --------------------------------
def vignette_mask(W: int, H: int, strength: float):
    if strength <= 0:
        return None
    y, x = np.ogrid[:H, :W]
    d = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
    m = 1.0 - strength * np.clip(d - 0.55, 0, 1) ** 1.5
    return np.repeat((m * 255).astype(np.uint8)[..., None], 3, axis=2)


class Grader:
    """Applies the template's look to every finished picture frame (before captions are drawn)."""

    def __init__(self, tpl: dict, W: int, H: int, seed: int = 0):
        g = tpl.get("grade", {})
        self.contrast = float(g.get("contrast", 1.0))
        self.brightness = float(g.get("brightness", 0.0))
        self.sat = float(g.get("saturation", 1.0))
        self.tint = np.array(list(g.get("tint", [0, 0, 0]))[::-1], np.int16)     # config is R,G,B; frames are B,G,R
        self.vig = vignette_mask(W, H, float(g.get("vignette", 0.0)))
        grain = float(g.get("grain", 0.0))
        self.noise = []
        if grain > 0:
            rng = np.random.default_rng(seed)
            self.noise = [(rng.normal(0, 1, (H, W, 1)) * 255 * grain).astype(np.int16) for _ in range(5)]
        self.letter = int(H * float(tpl.get("letterbox", 0.0)))

    def __call__(self, fr: np.ndarray, fi: int = 0) -> np.ndarray:
        if self.contrast != 1.0 or self.brightness != 0.0:
            fr = cv2.convertScaleAbs(fr, alpha=self.contrast, beta=self.brightness)
        if abs(self.sat - 1.0) > 0.01:
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
            hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * self.sat, 0, 255).astype(np.uint8)
            fr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if self.tint.any():
            fr = np.clip(fr.astype(np.int16) + self.tint, 0, 255).astype(np.uint8)
        if self.vig is not None:
            fr = cv2.multiply(fr, self.vig, scale=1 / 255.0)
        if self.noise:
            fr = np.clip(fr.astype(np.int16) + self.noise[fi % len(self.noise)], 0, 255).astype(np.uint8)
        if self.letter:
            fr[:self.letter] = 0
            fr[-self.letter:] = 0
        return fr


# ----------------------------- transitions ---------------------------------
def blend_frames(prev: np.ndarray, nxt: np.ndarray, u: float, kind: str) -> np.ndarray:
    """u runs 0..1 over the transition. kinds: fade, slide (push), dip (through black)."""
    u = min(max(u, 0.0), 1.0)
    if kind == "slide":
        W = prev.shape[1]
        s = int(smooth(u) * W)
        out = np.empty_like(prev)
        out[:, :W - s] = prev[:, s:]
        out[:, W - s:] = nxt[:, :s]
        return out
    if kind == "dip":
        if u < 0.5:
            return cv2.convertScaleAbs(prev, alpha=1 - 2 * u)
        return cv2.convertScaleAbs(nxt, alpha=2 * u - 1)
    a = smooth(u)
    return cv2.addWeighted(prev, 1 - a, nxt, a, 0)
