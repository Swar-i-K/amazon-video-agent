"""Shared helpers."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list, **kw) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{r.stderr[-1500:]}")
    return r


def ffprobe(path) -> dict:
    r = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    return json.loads(r.stdout)


def media_duration(path) -> float:
    return float(ffprobe(path)["format"]["duration"])


def video_info(path) -> dict:
    info = ffprobe(path)
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = (vs.get("avg_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    dur = float(info["format"].get("duration") or vs.get("duration") or 0)
    rot = 0
    for sd in vs.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = rot or int((vs.get("tags") or {}).get("rotate", 0) or 0)
    w, h = int(vs["width"]), int(vs["height"])
    if abs(rot) % 180 == 90:  # OpenCV applies rotation metadata automatically
        w, h = h, w
    return {"width": w, "height": h, "fps": fps or 30.0, "duration": dur}


def sha1_file(path, limit: int = 4_000_000) -> str:
    h = hashlib.sha1()
    p = Path(path)
    h.update(str(p.stat().st_size).encode())
    with open(p, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()[:16]


def sha1_text(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:20]


def extract_json(text: str):
    """Pull the first JSON object/array out of an LLM reply (tolerates code fences)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("No JSON found in model reply")
    obj, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    return obj


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9'’\-]+", text))


def fmt_ts(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"
