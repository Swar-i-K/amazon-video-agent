"""TV-commercial style product shots from a plain studio photo. CPU only, no AI model, free in GitHub Actions.

How it works: if the photo has a plain backdrop, the product is cut out, placed on a clean styled backdrop with a soft
contact shadow and floor reflection, then gently moved (push, pull, float, rise) while a light sweep glides across it.
Photos without a clean backdrop are skipped and use the normal pan/zoom instead.
"""
from __future__ import annotations

import cv2
import numpy as np

SWEEP_ANGLE = np.deg2rad(28)


def smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


# ----------------------------- cutout --------------------------------------
def cutout_mask(src: np.ndarray, max_side: int = 1400):
    """Return (src_resized, mask float32 0..1) for a plain-backdrop photo, or None if it is not suitable."""
    h, w = src.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        src = cv2.resize(src, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        h, w = src.shape[:2]
    b = max(3, min(h, w) // 60)
    strip = np.concatenate([src[:b].reshape(-1, 3), src[-b:].reshape(-1, 3),
                            src[:, :b].reshape(-1, 3), src[:, -b:].reshape(-1, 3)]).astype(np.float32)
    bg, std = strip.mean(0), float(strip.std(0).mean())
    if std > 9:           # backdrop is not uniform enough to cut out reliably
        return None
    diff = np.abs(src.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    thr = max(26, int(3 * std))
    m = (diff > thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    # fill holes (anything not reachable from the border)
    ff = m.copy()
    pad = cv2.copyMakeBorder(ff, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = pad.copy()
    cv2.floodFill(flood, np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8), (0, 0), 2)
    holes = (flood[1:-1, 1:-1] == 0)
    m = np.where(holes, 1, m).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0.004 * h * w]
    if not keep:
        return None
    m = np.isin(labels, keep).astype(np.uint8)
    frac = float(m.mean())
    if frac < 0.03 or frac > 0.80:
        return None
    ys, xs = np.where(m > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    touches = int(x0 <= 1) + int(y0 <= 1) + int(x1 >= w - 2) + int(y1 >= h - 2)
    if touches >= 3:      # product runs off the picture on most sides: cannot place it on a new backdrop
        return None
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.3)
    return src, np.clip(soft, 0, 1)


# ----------------------------- backdrop ------------------------------------
PALETTES = [  # (top, bottom) in BGR; light premium studio looks
    ((246, 246, 244), (214, 218, 222)),   # neutral studio
    ((232, 244, 252), (196, 214, 230)),   # warm cream
    ((250, 242, 232), (226, 208, 192)),   # cool blue-grey
    None,                                 # tinted from the product colour
]


def make_backdrop(W: int, H: int, variant: int, product_bgr: np.ndarray, spot_xy, floor_y: int) -> np.ndarray:
    pal = PALETTES[variant % len(PALETTES)]
    if pal is None:
        c = product_bgr.astype(np.float32)
        hsv = cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
        top = cv2.cvtColor(np.uint8([[[hsv[0], min(hsv[1], 255) * 0.18, 245]]]), cv2.COLOR_HSV2BGR)[0, 0]
        bot = cv2.cvtColor(np.uint8([[[hsv[0], min(hsv[1], 255) * 0.30, 205]]]), cv2.COLOR_HSV2BGR)[0, 0]
        pal = (tuple(int(x) for x in top), tuple(int(x) for x in bot))
    top, bot = np.array(pal[0], np.float32), np.array(pal[1], np.float32)
    t = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    bg = top * (1 - t) + bot * t
    bg = np.repeat(bg, W, axis=1)
    # soft spotlight behind the product
    y, x = np.ogrid[:H, :W]
    d = np.sqrt(((x - spot_xy[0]) / (W * 0.45)) ** 2 + ((y - spot_xy[1]) / (H * 0.55)) ** 2)
    glow = np.clip(1 - d, 0, 1)[..., None] ** 2
    bg = bg + glow * 22
    # table/floor: slightly darker below the horizon with a soft transition
    fl = np.clip((y - floor_y) / (H * 0.08), -1, 1)
    floor = (0.5 + 0.5 * fl)[..., None]
    bg = bg * (1 - 0.07 * floor)
    return np.clip(bg, 0, 255).astype(np.uint8)


# ----------------------------- the shot ------------------------------------
MOTIONS = [
    dict(scale=(0.93, 1.04), rot=(0.0, 0.0), dx=(0.0, 0.0), dy=(0.0, 0.0), bg=(-0.02, 0.02)),     # push in
    dict(scale=(1.07, 0.95), rot=(0.0, 0.0), dx=(0.0, 0.0), dy=(0.0, 0.0), bg=(0.02, -0.02)),     # pull back
    dict(scale=(1.0, 1.03), rot=(-3.0, 3.0), dx=(-0.02, 0.02), dy=(0.0, 0.0), bg=(0.02, -0.02)),  # gentle orbit
    dict(scale=(0.96, 1.02), rot=(0.5, -0.5), dx=(0.0, 0.0), dy=(0.04, -0.02), bg=(0.0, 0.0)),    # rise
]


class CinematicProduct:
    def __init__(self, src_bgr: np.ndarray, W: int, H: int, variant: int = 0, seed: int = 0):
        res = cutout_mask(src_bgr)
        if res is None:
            raise ValueError("no clean cutout")
        img, mask = res
        ys, xs = np.where(mask > 0.5)
        pad = 6
        y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad + 1)
        crop, a = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        # scale product to sit nicely in frame
        ph, pw = crop.shape[:2]
        k = min(0.66 * H / ph, 0.66 * W / pw, 2.0)
        sw, sh = max(8, int(pw * k)), max(8, int(ph * k))
        crop = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        a = cv2.resize(a, (sw, sh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        self.W, self.H, self.variant = W, H, variant
        self.motion = MOTIONS[variant % len(MOTIONS)]
        self.sw, self.sh = sw, sh
        # product colour for tinted backdrops
        pc = crop[a > 0.8].mean(axis=0) if (a > 0.8).any() else np.array([200, 200, 200], np.float32)
        # sprite = product + faded reflection beneath (moves as one piece)
        refl_h = int(sh * 0.45)
        refl = cv2.flip(crop, 0)[:refl_h]
        ra = cv2.flip(a, 0)[:refl_h] * np.linspace(0.30, 0.0, refl_h, dtype=np.float32)[:, None]
        self.spr_h = sh + refl_h
        spr = np.zeros((self.spr_h, sw, 3), np.float32)
        spa = np.zeros((self.spr_h, sw), np.float32)
        spr[:sh], spa[:sh] = crop, a
        spr[sh:], spa[sh:] = refl, ra
        self.spr_bgr, self.spr_a = spr, spa
        self.prod_a = a  # for the light sweep (product area only)
        # layout: product centred slightly above middle, base on the "table"
        self.cx, self.base_y = W // 2, int(H * 0.80)
        self.floor_y = self.base_y
        bg = make_backdrop(int(W * 1.12), int(H * 1.12), variant, pc, (int(W * 1.12 * 0.5), int(H * 1.12 * 0.52)), int(self.floor_y * 1.12))
        self.bg = bg
        # contact shadow (precomputed, scales with product width)
        shw, shh = int(sw * 1.05), max(10, int(sw * 0.10))
        sh_img = np.zeros((shh * 4, shw + 40), np.float32)
        cv2.ellipse(sh_img, (sh_img.shape[1] // 2, sh_img.shape[0] // 2), (shw // 2, shh), 0, 0, 360, 1.0, -1)
        self.shadow = cv2.GaussianBlur(sh_img, (0, 0), shh * 0.9) * 0.55
        self.sweep_phase = 0.2 + 0.1 * ((seed + variant) % 3)
        yy, xx = np.mgrid[0:self.sh, 0:self.sw].astype(np.float32)
        self.proj = xx * np.cos(SWEEP_ANGLE) + yy * np.sin(SWEEP_ANGLE)
        self.proj_lo, self.proj_hi = float(self.proj.min()), float(self.proj.max())
        self.sweep_gain = (self.prod_a * 70.0)[..., None].astype(np.float32)

    @staticmethod
    def _roi(M, w, h, W, H, extra=2):
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        out = pts @ M[:, :2].T + M[:, 2]
        x0, y0 = np.floor(out.min(0)).astype(int) - extra
        x1, y1 = np.ceil(out.max(0)).astype(int) + extra
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

    def render(self, t: float, dur: float) -> np.ndarray:
        W, H = self.W, self.H
        m = self.motion
        u = smooth(t / max(dur, 1e-3))
        lerp = lambda ab: ab[0] + (ab[1] - ab[0]) * u
        sc, rot = lerp(m["scale"]), lerp(m["rot"])
        dx, dy = lerp(m["dx"]) * W, lerp(m["dy"]) * H
        bgoff = lerp(m["bg"])
        # backdrop with slight parallax (moves opposite to, and less than, the product)
        bh, bw = self.bg.shape[:2]
        ox = min(max(int((bw - W) / 2 + bgoff * W), 0), bw - W)
        oy = (bh - H) // 2
        frame = np.ascontiguousarray(self.bg[oy:oy + H, ox:ox + W])

        base_x, base_y = self.cx + dx, self.base_y + dy
        c, s = np.cos(np.deg2rad(rot)) * sc, np.sin(np.deg2rad(rot)) * sc
        pivot = (self.sw / 2, self.sh)             # base of the product inside the sprite
        M = np.float32([[c, s, base_x - (c * pivot[0] + s * pivot[1])],
                        [-s, c, base_y - (-s * pivot[0] + c * pivot[1])]])

        # contact shadow (region only)
        sx, sy = int(base_x - self.shadow.shape[1] * sc / 2), int(base_y - self.shadow.shape[0] * sc / 2 + 4)
        sM = np.float32([[sc, 0, sx], [0, sc, sy]])
        r = self._roi(sM, self.shadow.shape[1], self.shadow.shape[0], W, H)
        if r:
            x0, y0, x1, y1 = r
            sM2 = sM.copy(); sM2[0, 2] -= x0; sM2[1, 2] -= y0
            shw = cv2.warpAffine(self.shadow, sM2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
            reg = frame[y0:y1, x0:x1].astype(np.float32) * (1 - shw[..., None] * 0.9)
            frame[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)

        # light sweep on the product part of the sprite, then product + reflection
        spr = self.spr_bgr
        sweep_u = (u - self.sweep_phase) / 0.55
        if 0.0 <= sweep_u <= 1.0:
            pos = self.proj_lo + (self.proj_hi - self.proj_lo) * (-0.15 + 1.3 * sweep_u)
            band = np.exp(-((self.proj - pos) / (0.07 * (self.proj_hi - self.proj_lo))) ** 2)
            spr = self.spr_bgr.copy()
            spr[:self.sh] += band[..., None] * self.sweep_gain
        pm = np.dstack([np.clip(spr, 0, 255) * self.spr_a[..., None], self.spr_a]).astype(np.float32)
        r = self._roi(M, self.sw, self.spr_h, W, H)
        if r:
            x0, y0, x1, y1 = r
            M2 = M.copy(); M2[0, 2] -= x0; M2[1, 2] -= y0
            wp = cv2.warpAffine(pm, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            reg = wp[..., :3] + frame[y0:y1, x0:x1].astype(np.float32) * (1 - wp[..., 3:4])
            frame[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)
        return frame


def can_cutout(src_bgr: np.ndarray) -> bool:
    try:
        return cutout_mask(src_bgr) is not None
    except Exception:
        return False
