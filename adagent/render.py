"""Turn the user's own images/clips into a polished faceless product-ad video.

For every narration line we plan 1-3 shots. Photos get smooth pan/zoom moves with different framings on
each reuse; clips are cut at different in-points; everything sits on a clean (studio or blurred) background,
cross-fades between shots, and carries captions + callouts.
"""
from __future__ import annotations

import random
import re
import subprocess
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .assets import load_image_bgr
from .common import log, video_info
from .overlay import Overlay

STAGE = 1.2  # photos are composed 20% larger than the output so zooms stay sharp


def smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


# ----------------------------- image composition ----------------------------
def cover(img: np.ndarray, W: int, H: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(W / w, H / h)
    r = cv2.resize(img, (int(np.ceil(w * s)), int(np.ceil(h * s))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    y0, x0 = (r.shape[0] - H) // 2, (r.shape[1] - W) // 2
    return np.ascontiguousarray(r[y0:y0 + H, x0:x0 + W])


def border_stats(img: np.ndarray):
    h, w = img.shape[:2]
    b = max(2, min(h, w) // 50)
    strip = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                            img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)]).astype(np.float32)
    return strip.mean(0), float(strip.std(0).mean())


def soft_bg(img: np.ndarray, W: int, H: int) -> np.ndarray:
    small = cover(img, 240, max(8, int(240 * H / W)))
    small = cv2.GaussianBlur(small, (0, 0), 5)
    return (cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) * 0.55).astype(np.uint8)


def compose(src: np.ndarray, Sw: int, Sh: int, studio: bool | None = None) -> np.ndarray:
    h, w = src.shape[:2]
    if abs((w / h) / (Sw / Sh) - 1) < 0.14:
        return cover(src, Sw, Sh)
    mean, std = border_stats(src)
    if studio is None:
        studio = bool(mean.min() > 222 and std < 10)
    s = min(int(Sw * 0.82) / w, int(Sh * 0.90) / h, 2.2)
    fw, fh = max(2, int(w * s)), max(2, int(h * s))
    fg = cv2.resize(src, (fw, fh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    x0, y0 = (Sw - fw) // 2, (Sh - fh) // 2
    if studio:
        bg = np.ones((Sh, Sw, 3), np.float32) * mean  # flat colour = the photo's own backdrop, so no seam shows
        m = np.zeros((fh, fw), np.float32)
        m[6:-6, 6:-6] = 1.0
        m = cv2.GaussianBlur(m, (0, 0), 4)[..., None]
        reg = bg[y0:y0 + fh, x0:x0 + fw]
        bg[y0:y0 + fh, x0:x0 + fw] = reg * (1 - m) + fg.astype(np.float32) * m
        return bg.astype(np.uint8)
    bg = soft_bg(src, Sw, Sh).astype(np.float32)
    sh = np.zeros((Sh, Sw), np.float32)
    sh[y0 + 14:y0 + 14 + fh, x0 + 14:x0 + 14 + fw] = 0.55
    sh = cv2.GaussianBlur(sh, (0, 0), 18)[..., None]
    bg = bg * (1 - sh)
    bg[y0:y0 + fh, x0:x0 + fw] = fg
    return np.clip(bg, 0, 255).astype(np.uint8)


def camera(stage: np.ndarray, W: int, H: int, z: float, fx: float, fy: float) -> np.ndarray:
    Sh, Sw = stage.shape[:2]
    hw, hh = Sw / (2 * z), Sh / (2 * z)
    cx, cy = hw + fx * (Sw - 2 * hw), hh + fy * (Sh - 2 * hh)
    k = z * W / Sw
    M = np.float32([[k, 0, W / 2 - k * cx], [0, k, H / 2 - k * cy]])
    return cv2.warpAffine(stage, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


MOVES = [
    dict(z=(1.00, 1.12), c=((.5, .5), (.5, .5))),     # slow push in
    dict(z=(1.14, 1.00), c=((.5, .5), (.5, .5))),     # pull back
    dict(z=(1.15, 1.15), c=((.15, .5), (.85, .5))),   # pan right
    dict(z=(1.15, 1.15), c=((.85, .5), (.15, .5))),   # pan left
    dict(z=(1.08, 1.24), c=((.30, .30), (.55, .50))), # drift into upper-left detail
    dict(z=(1.08, 1.24), c=((.70, .65), (.45, .45))), # drift into lower-right detail
    dict(z=(1.18, 1.18), c=((.5, .15), (.5, .85))),   # tilt down
    dict(z=(1.18, 1.18), c=((.5, .85), (.5, .15))),   # tilt up
]


def limit_move(move: dict, w_src: int, W: int, contained: bool) -> dict:
    """Do not zoom far into small, low-resolution photos."""
    up = (0.82 if contained else 1.0) * W / max(1, w_src)
    z_cap = min(1.35, max(1.04, 1.6 / max(up, 1e-3)))
    f = min(1.0, (z_cap - 1.0) / 0.24)
    z0, z1 = move["z"]
    return dict(z=(1 + (z0 - 1) * f, 1 + (z1 - 1) * f), c=move["c"])


# ----------------------------- shots ---------------------------------------
class ClipReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.pos, self.last = -1, None

    def frame_at(self, sec: float):
        idx = int(sec * self.fps)
        if self.n > 0:
            idx %= self.n
        if idx == self.pos and self.last is not None:
            return self.last
        if idx < self.pos or idx - self.pos > 20:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.pos = idx - 1
        ok = True
        while self.pos < idx:
            ok = self.cap.grab()
            self.pos += 1
            if not ok:
                break
        ok, fr = self.cap.retrieve()
        if ok:
            self.last = fr
        return self.last

    def close(self):
        self.cap.release()


@dataclass
class Shot:
    start: float
    dur: float
    asset: dict
    path: Path
    W: int
    H: int
    move: dict = field(default_factory=dict)
    clip_in: float = 0.0
    speed: float = 1.0
    _stage: np.ndarray | None = None
    _reader: ClipReader | None = None
    _studio: bool | None = None

    def render(self, t: float) -> np.ndarray:
        W, H = self.W, self.H
        u = t / max(self.dur, 1e-3)
        e = smooth(u)
        z0, z1 = self.move["z"]
        (ax, ay), (bx, by) = self.move["c"]
        z, fx, fy = z0 + (z1 - z0) * e, ax + (bx - ax) * e, ay + (by - ay) * e
        if self.asset["kind"] == "image":
            if self._stage is None:
                self._stage = compose(load_image_bgr(self.path), int(W * STAGE), int(H * STAGE))
            return camera(self._stage, W, H, z, fx, fy)
        if self._reader is None:
            self._reader = ClipReader(self.path)
        fr = self._reader.frame_at(self.clip_in + t * self.speed)
        if fr is None:
            return np.zeros((H, W, 3), np.uint8)
        if self._studio is None:
            m, s = border_stats(fr)
            self._studio = bool(m.min() > 222 and s < 10)
        stage = compose(fr, W, H, self._studio)
        return camera(stage, W, H, 1.0 + (z - 1.0) * 0.4, 0.5, 0.5) if z != 1.0 else stage

    def release(self):
        self._stage = None
        if self._reader:
            self._reader.close()
            self._reader = None


def plan_shots(rows, total: float, entries: list[dict], project_assets: Path, cfg: dict, seed: str) -> list[Shot]:
    v = cfg["video"]
    W, H = v["width"], v["height"]
    rng = random.Random(seed)
    usable = [e for e in entries if e["usable"]]
    if not usable:
        raise SystemExit("No usable assets. Add product photos/clips to the project's assets/ folder.")
    by_id = {e["id"]: e for e in usable}
    usage, view_count, cursor = Counter(), Counter(), Counter()
    first_move: dict = {}
    recent: deque = deque(maxlen=3)
    last_kind = None
    shots: list[Shot] = []
    target = v["shot_seconds"]

    def tokens(s):
        return set(re.findall(r"[a-z]{4,}", s.lower()))

    bounds = [r.start for r in rows] + [total]
    bounds[0] = 0.0  # first shot starts at t=0 (lead-in)
    for i, r in enumerate(rows):
        span = bounds[i + 1] - bounds[i]
        n = max(1, min(int(span // 3.4), round(span / target)))
        per = span / n
        want = [by_id[a] for a in r.seg.get("asset_ids", []) if a in by_id]
        txt = tokens(r.seg["text"])
        for k in range(n):
            want_ids = {w["id"] for w in want}

            def score(e):
                sc = 0.0
                if e["id"] in want_ids:
                    sc += 1.6
                sc += 0.25 * len(txt & tokens(e["description"])) / 3
                sc -= 0.45 * usage[e["id"]] / max(1, len(usable) ** 0.5)
                if e["id"] in recent:
                    sc -= 3.0
                if last_kind and e["kind"] != last_kind:
                    sc += 0.3
                return sc + rng.random() * 0.05

            e = max(usable, key=score)
            path = project_assets / e["file"]
            shot = Shot(start=bounds[i] + k * per, dur=per, asset=e, path=path, W=W, H=H)
            if e["kind"] == "image":
                if e["id"] not in first_move:
                    first_move[e["id"]] = rng.randrange(len(MOVES))
                mv = MOVES[(first_move[e["id"]] + 3 * view_count[e["id"]]) % len(MOVES)]  # 3 is coprime with 8: every reuse differs
                contained = abs((e["width"] / e["height"]) / (W / H) - 1) >= 0.14
                shot.move = limit_move(mv, e["width"], W, contained)
                view_count[e["id"]] += 1
            else:
                need = per + v["crossfade"] + 0.2
                speed = 1.0 if e["duration"] >= need else max(0.6, e["duration"] / need)
                room = max(0.0, e["duration"] - need * speed)
                shot.speed = speed
                shot.clip_in = (cursor[e["id"]] % (room + 0.01)) if room > 0.3 else 0.0
                cursor[e["id"]] += need * speed + 0.7
                shot.move = dict(z=(1.0, 1.0 + 0.05 * rng.random()), c=((.5, .5), (.5, .5)))
            usage[e["id"]] += 1
            recent.append(e["id"])
            last_kind = e["kind"]
            shots.append(shot)
    return shots


# ----------------------------- main render ----------------------------------
def vignette_mask(W: int, H: int) -> np.ndarray:
    y, x = np.ogrid[:H, :W]
    d = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
    m = 1.0 - 0.28 * np.clip(d - 0.55, 0, 1) ** 1.5
    return np.repeat((m * 255).astype(np.uint8)[..., None], 3, axis=2)


def render_video(shots: list[Shot], overlays: list[Overlay], total: float, cfg: dict, out_path: Path,
                 log_path: Path, max_seconds: float | None = None) -> None:
    v = cfg["video"]
    W, H, fps, xf = v["width"], v["height"], v["fps"], v["crossfade"]
    n_frames = int(round((max_seconds or total) * fps))
    vig = vignette_mask(W, H)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", v["preset"], "-crf", str(v["crf"]),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=lf)
        k = 0
        tick = max(1, n_frames // 10)
        try:
            for fi in range(n_frames):
                t = fi / fps
                while k + 1 < len(shots) and shots[k + 1].start <= t:
                    k += 1
                    for old in shots[:max(0, k - 2)]:
                        old.release()
                sh = shots[k]
                fr = sh.render(t - sh.start)
                if k > 0 and t - sh.start < xf:
                    pv = shots[k - 1]
                    a = smooth((t - sh.start) / xf)
                    fr = cv2.addWeighted(pv.render(t - pv.start), 1 - a, fr, a, 0)
                fr = cv2.convertScaleAbs(fr, alpha=1.04, beta=-4)       # gentle contrast
                fr = cv2.multiply(fr, vig, scale=1 / 255.0)             # soft vignette
                for ov in overlays:
                    ov.apply(fr, t)
                proc.stdin.write(np.ascontiguousarray(fr).tobytes())
                if fi % tick == 0:
                    log(f"  rendering {100 * fi // n_frames}%")
        finally:
            proc.stdin.close()
            code = proc.wait()
            for s in shots:
                s.release()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed, see {log_path}")
