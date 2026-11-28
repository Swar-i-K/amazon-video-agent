"""Free, local pre-scan of every photo and clip (no AI, no Groq tokens).

For each file we measure: size, sharpness (edge strength), exposure, a near-duplicate fingerprint, a colour fingerprint, and for clips
the best 5-second moments (sharp + moving). Results are cached by file content in
projects/<ASIN>/.cache/prescan.json, so each file is scanned only once.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .common import log, sha1_file, video_info

SCAN_VERSION = 3          # bump when the measurements change (old cache entries are then re-measured)
EDGE_GOOD = 100.0         # edge strength of a clearly sharp photo (mean of the strongest 0.1% |Laplacian| values at 640 px)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def _hist(bgr: np.ndarray) -> list[float]:
    hsv = cv2.cvtColor(cv2.resize(bgr, (96, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    hs = cv2.calcHist([hsv], [0, 1], None, [8, 3], [0, 180, 0, 256]).flatten()
    v = cv2.calcHist([hsv], [2], None, [4], [0, 256]).flatten()
    h = np.concatenate([hs / max(hs.sum(), 1), v / max(v.sum(), 1)])
    return [round(float(x), 4) for x in h]


def _edge(gray: np.ndarray) -> float:
    """Edge strength = mean of the strongest 0.1% edge responses. Photos that are mostly plain backdrop (product on white) have few edge
    pixels, so a variance or 99th-percentile measure would wrongly call them blurry; the strongest edges still tell sharp from soft."""
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F)).ravel()
    k = max(10, lap.size // 1000)
    return float(np.mean(np.partition(lap, -k)[-k:]))


def _small(bgr: np.ndarray, side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = side / max(h, w)
    return cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else bgr


def _scan_image(path: Path) -> dict:
    from .assets import load_image_bgr
    img = load_image_bgr(path)
    h, w = img.shape[:2]
    small = _small(img, 640)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sharp = _edge(gray)
    mean, std = float(gray.mean()), float(gray.std())
    sharp_n = min(1.0, sharp / EDGE_GOOD)
    res_n = min(1.0, min(h, w) / 1080.0)
    expo_n = 1.0 if 40 <= mean <= 238 else 0.3
    return {"kind": "image", "w": w, "h": h, "dur": 0.0, "sharp": round(sharp, 2), "mean": round(mean, 1), "std": round(std, 1),
            "dhash": format(_dhash(gray), "016x"), "hist": _hist(small),
            "quality": round(0.6 * sharp_n + 0.25 * res_n + 0.15 * expo_n, 4), "windows": []}


def _scan_clip(path: Path) -> dict:
    vi = video_info(path)
    dur, fps = float(vi["duration"]), float(vi["fps"] or 25.0)
    cap = cv2.VideoCapture(str(path))
    step_frames = max(1, int(round(fps)))                 # one sample per second
    total = int(min(dur, 180.0) * fps)
    times, edges, motion, frames_small = [], [], [], []
    prev, fi = None, 0
    while fi < total:
        if not cap.grab():
            break
        if fi % step_frames == 0:
            ok, fr = cap.retrieve()
            if ok and fr is not None:
                g = cv2.cvtColor(_small(fr, 160), cv2.COLOR_BGR2GRAY)
                times.append(fi / fps)
                edges.append(_edge(g))
                motion.append(float(np.abs(g.astype(np.int16) - prev.astype(np.int16)).mean()) if prev is not None and prev.shape == g.shape else 0.0)
                frames_small.append(_small(fr, 320))
                prev = g
        fi += 1
    cap.release()
    if not times:
        return {"kind": "clip", "w": vi["width"], "h": vi["height"], "dur": dur, "sharp": 0.0, "mean": 0.0, "std": 0.0, "dhash": "0" * 16,
                "hist": [0.0] * 28, "quality": 0.0, "windows": [], "unreadable": True}
    sharp_n = [min(1.0, e / (EDGE_GOOD * 0.7)) for e in edges]
    mot_n = [min(1.0, m / 12.0) if m < 30 else 0.3 for m in motion]      # very high motion = shaky, not "dynamic"
    L = min(5.0, max(1.0, dur))
    cands = []
    t = 0.0
    while t + min(L, dur) <= dur + 0.01:
        idx = [i for i, tt in enumerate(times) if t <= tt < t + L] or [min(range(len(times)), key=lambda i: abs(times[i] - t))]
        sc = 0.6 * float(np.mean([sharp_n[i] for i in idx])) + 0.4 * float(np.mean([mot_n[i] for i in idx]))
        cands.append((sc, round(t, 2)))
        t += 1.0
    cands.sort(reverse=True)
    wins: list[dict] = []
    for sc, st in cands:
        if all(abs(st - w["start"]) >= L * 0.8 for w in wins):
            wins.append({"start": st, "score": round(sc, 4)})
        if len(wins) >= 8:
            break
    gmid = cv2.cvtColor(_small(frames_small[len(frames_small) // 2], 160), cv2.COLOR_BGR2GRAY)
    q = float(np.mean([w["score"] for w in wins[:3]])) if wins else 0.0
    res_n = min(1.0, min(vi["width"], vi["height"]) / 720.0)
    return {"kind": "clip", "w": vi["width"], "h": vi["height"], "dur": round(dur, 2), "sharp": round(float(np.mean(edges)), 2),
            "mean": round(float(gmid.mean()), 1), "std": round(float(gmid.std()), 1), "dhash": format(_dhash(gmid), "016x"),
            "hist": _hist(frames_small[len(frames_small) // 2]), "quality": round(0.8 * q + 0.2 * res_n, 4), "windows": wins}


def scan_all(project, files: list[Path], cfg: dict | None = None) -> dict[str, dict]:
    """{file name: measurements}. Cached by file content; a file that cannot be read gets {'unreadable': True}."""
    from .assets import IMG_EXT
    cache_file = project.cache_dir / "prescan.json"
    cache: dict = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except Exception:
            cache = {}
    out: dict[str, dict] = {}
    new = 0
    for p in files:
        h = sha1_file(p)
        c = cache.get(h)
        if c and c.get("v") == SCAN_VERSION:
            out[p.name] = c
            continue
        try:
            feat = _scan_image(p) if p.suffix.lower() in IMG_EXT else _scan_clip(p)
        except Exception as e:
            log(f"  could not scan {p.name}: {e}")
            feat = {"kind": "image" if p.suffix.lower() in IMG_EXT else "clip", "unreadable": True, "quality": 0.0, "w": 0, "h": 0,
                    "dur": 0.0, "windows": [], "hist": [0.0] * 28, "dhash": "0" * 16, "sharp": 0.0, "mean": 0.0, "std": 0.0}
        feat["v"] = SCAN_VERSION
        cache[h] = feat
        out[p.name] = feat
        new += 1
        if new % 10 == 0:
            cache_file.write_text(json.dumps(cache))
    if new:
        cache_file.write_text(json.dumps(cache))
        log(f"  pre-scanned {new} new file(s) locally (free; no AI used)")
    return out
