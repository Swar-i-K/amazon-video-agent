"""Lay narration lines on a timeline, build the voice track, enforce the 5-9 minute window."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scriptgen import SECTION_LABELS
from .tts import SR, _write_wav
from pathlib import Path


@dataclass
class Row:
    start: float
    dur: float
    pad: float
    seg: dict


@dataclass
class Timeline:
    rows: list
    total: float
    voice: np.ndarray
    voice_end: float


def _silence(sec: float) -> np.ndarray:
    return np.zeros(int(round(sec * SR)), dtype=np.int16)


def _layout(segments, arrays, cfg, pause_scale: float):
    ln, v = cfg["length"], cfg["voice"]
    t = ln["lead_in"]
    rows, chunks = [], [_silence(ln["lead_in"])]
    for i, (s, a) in enumerate(zip(segments, arrays)):
        dur = len(a) / SR
        nxt = segments[i + 1]["section"] if i + 1 < len(segments) else None
        pad = v["pause_sentence"] + (v["pause_section"] if nxt and nxt != s["section"] else 0.0)
        pad *= pause_scale
        rows.append(Row(t, dur, pad, s))
        chunks += [a, _silence(pad)]
        t += dur + pad
    voice_end = rows[-1].start + rows[-1].dur
    total = voice_end + ln["tail"]
    return rows, chunks, voice_end, total


def build_timeline(segments, arrays, cfg, enforce: bool = True) -> Timeline:
    ln = cfg["length"]
    lo, hi = ln["min_minutes"] * 60, ln["max_minutes"] * 60
    scale = 1.0
    rows, chunks, voice_end, total = _layout(segments, arrays, cfg, scale)
    # Gently stretch / squeeze the pauses (never the voice) if we are just outside the window.
    for _ in range(12 if enforce else 0):
        if lo + 2 <= total <= hi - 2:
            break
        scale = scale * (1.12 if total < lo + 2 else 0.88)
        scale = min(max(scale, 0.5), 2.2)
        rows, chunks, voice_end, total = _layout(segments, arrays, cfg, scale)
        if scale in (0.5, 2.2):
            break
    if enforce and not (lo <= total <= hi):
        raise SystemExit(
            f"Final length is {total / 60:.2f} min, outside the required {ln['min_minutes']}-{ln['max_minutes']} min. "
            "Edit the script (add/remove lines) and render again. Cached voice lines are reused, so edits are cheap.")
    voice = np.concatenate(chunks + [_silence(ln["tail"])])
    return Timeline(rows, total, voice, voice_end)


def write_voice(tl: Timeline, path) -> None:
    _write_wav(Path(path), tl.voice)


def chapters(tl: Timeline) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    for r in tl.rows:
        label = SECTION_LABELS.get(r.seg["section"], r.seg["section"].replace("_", " ").title())
        if not out:
            out.append((0.0, label))
        elif label != out[-1][1] and r.start - out[-1][0] >= 20:
            out.append((r.start, label))
    return out if len(out) >= 3 else []
