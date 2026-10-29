"""Scan the project's assets, describe them, and screen out anything unusable.

Screening rules (the video must be faceless and must not show Amazon UI):
  * any human face (vision model + OpenCV backup)  -> excluded
  * visible prices / star ratings / review text / buy buttons -> excluded
  * not actually showing the product                -> excluded
Override a false positive by listing the filename under  assets_allow:  in facts.yaml.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .common import extract_json, log, sha1_file, video_info
from .llm import DailyLimitError, ask
from .project import Project

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VID_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


def load_image_bgr(path) -> np.ndarray:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        im = im.convert("RGBA")
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def sample_frames(path, n: int = 4) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    for i in range(n):
        pos = int(total * (0.1 + 0.8 * i / max(1, n - 1))) if total else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    return frames


def contact_sheet(frames: list[np.ndarray], tile_w: int = 640) -> np.ndarray:
    tiles = []
    for f in frames:
        h, w = f.shape[:2]
        s = tile_w / w
        tiles.append(cv2.resize(f, (tile_w, int(h * s)), interpolation=cv2.INTER_AREA))
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in tiles]
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    return np.vstack(rows)


def as_bool(v, default: bool = False) -> bool:
    """Models sometimes answer "false"/"no" as text; bool("false") would wrongly be True."""
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return v != 0
    t = str(v).strip().lower()
    if t in ("true", "yes", "y", "1"):
        return True
    if t in ("false", "no", "n", "0", "none", "null", ""):
        return False
    return default


_FACE = None


def haar_has_face(frames: list[np.ndarray]) -> bool:
    """OpenCV's classic face detector, used as a backup to the vision model. If this OpenCV build lacks it, skip quietly."""
    global _FACE
    if _FACE is None:
        try:
            det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            _FACE = det if not det.empty() else False
        except Exception:
            _FACE = False
        if _FACE is False:
            log("  (OpenCV face-detector backup is not available in this build; relying on the AI image check only. "
                "Look through your footage yourself for faces.)")
    if _FACE is False:
        return False
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        s = 800 / max(g.shape)
        if s < 1:
            g = cv2.resize(g, None, fx=s, fy=s)
        faces = _FACE.detectMultiScale(g, scaleFactor=1.1, minNeighbors=7, minSize=(48, 48))
        if len(faces):
            return True
    return False


VISION_SYSTEM = (
    "You are a careful assistant for a video editor. You inspect product photos/video frames and answer ONLY with JSON."
)

VISION_PROMPT = """Inspect {what}. Return JSON with exactly these keys:
{{"description": "<=25 words, literal description of what is visible: product, angle, setting, notable details>",
  "shows_product": true/false,
  "has_human_face": true/false,   // ANY human face, even partial, small, in a photo on packaging, or a reflection
  "has_price_or_rating_ui": true/false,  // visible price, star rating, review text, Amazon page UI, buy button, badge like 'Best Seller'
  "has_third_party_logo_or_watermark": true/false}}
Do not guess about things you cannot see."""


def describe(cfg: dict, frames: list[np.ndarray], kind: str) -> dict:
    img = frames[0] if kind == "image" else contact_sheet(frames)
    what = "this product photo" if kind == "image" else "this contact sheet of 4 frames from one product video clip"
    last = None
    for _ in range(2):
        try:
            txt = ask(cfg["llm"]["vision_model"], VISION_SYSTEM, VISION_PROMPT.format(what=what),
                      images=[img], max_tokens=400, temperature=0.0)
            return extract_json(txt)
        except DailyLimitError:
            raise
        except Exception as e:
            last = e
    raise RuntimeError(f"Vision check failed: {last}")


def build_index(project: Project, cfg: dict, vision: bool = True) -> list[dict]:
    """Return a list of asset dicts (usable and unusable) and cache it. Progress is saved after every photo, so a stopped
    run never has to re-check what was already checked."""
    if vision and not cfg.get("llm", {}).get("vision_check", True):
        log("  (AI photo check is turned off in config.yaml: llm.vision_check. Faces are only checked locally; "
            "look through your photos yourself for faces, prices and Amazon page elements.)")
        vision = False
    cache_file = project.cache_dir / "assets_index.json"
    cache = {}
    if cache_file.exists():
        try:
            cache = {e["hash"]: e for e in json.loads(cache_file.read_text())}
        except Exception:
            cache = {}
    allow = {str(x).lower() for x in (project.facts.get("assets_allow") or [])}
    files = sorted(p for p in project.assets_dir.glob("*") if p.suffix.lower() in IMG_EXT | VID_EXT)
    entries: list[dict] = []

    def save() -> None:
        merged = dict(cache)
        merged.update({x["hash"]: x for x in entries if "hash" in x})
        cache_file.write_text(json.dumps(list(merged.values()), indent=1))

    n_new = 0
    for i, p in enumerate(files, 1):
        h = sha1_file(p)
        fresh = False
        if h in cache and (cache[h].get("vision") or not vision):
            e = dict(cache[h])
            e["file"], e["id"] = p.name, f"A{i:02d}"
        else:
            fresh = True
            kind = "image" if p.suffix.lower() in IMG_EXT else "clip"
            e = {"id": f"A{i:02d}", "file": p.name, "hash": h, "kind": kind, "vision": vision}
            if kind == "image":
                img = load_image_bgr(p)
                e["width"], e["height"] = img.shape[1], img.shape[0]
                frames = [img]
                e["duration"] = 0.0
            else:
                vi = video_info(p)
                e.update(width=vi["width"], height=vi["height"], duration=vi["duration"], fps=vi["fps"])
                frames = sample_frames(p)
            if not frames:
                e.update(usable=False, reason="could not be read", description=p.stem)
                entries.append(e)
                continue
            e["haar_face"] = haar_has_face(frames)
            if vision:
                log(f"  describing {p.name} ...")
                try:
                    v = describe(cfg, frames, kind)
                except DailyLimitError as err:
                    save()
                    raise SystemExit(
                        f"{err}\n"
                        f"Your progress is saved: {len(entries)} of {len(files)} files in projects/{project.asin}/assets were already checked "
                        "and will not be checked again.\n"
                        "What you can do: (1) run again tomorrow, (2) add models under llm.vision_fallbacks in config.yaml (names from "
                        "your Groq account's model list), (3) turn the AI photo check off with  llm: vision_check: false  in config.yaml "
                        "(then check your own photos for faces and Amazon page elements), or (4) upgrade the Groq tier.")
                e["description"] = str(v.get("description", p.stem))
                e["shows_product"] = as_bool(v.get("shows_product"), True)
                e["llm_face"] = as_bool(v.get("has_human_face"), False)
                e["price_ui"] = as_bool(v.get("has_price_or_rating_ui"), False)
                e["logo_watermark"] = as_bool(v.get("has_third_party_logo_or_watermark"), False)
            else:
                e.update(description=p.stem.replace("_", " ").replace("-", " "), shows_product=True,
                         llm_face=False, price_ui=False, logo_watermark=False)
        reasons = []
        if (e.get("llm_face") or e.get("haar_face")) and p.name.lower() not in allow:
            reasons.append("shows a human face")
        if e.get("price_ui") and p.name.lower() not in allow:
            reasons.append("shows price/rating/Amazon UI")
        if not e.get("shows_product", True) and p.name.lower() not in allow and not p.name.lower().startswith("ai_"):
            reasons.append("does not show the product")
        e["usable"] = not reasons
        e["reason"] = "; ".join(reasons)
        entries.append(e)
        if fresh:
            n_new += 1
            save()
    save()
    return entries


def coverage_report(entries: list[dict], target_seconds: float) -> tuple[float, list[str]]:
    """Estimate how much *distinct* picture the assets provide vs the video length."""
    notes = []
    usable = [e for e in entries if e["usable"]]
    imgs = [e for e in usable if e["kind"] == "image"]
    clips = [e for e in usable if e["kind"] == "clip"]
    unique = sum(c["duration"] for c in clips) + 3 * 4.5 * len(imgs)  # ~3 distinct framings per photo
    cov = unique / max(1.0, target_seconds)
    notes.append(f"{len(imgs)} usable photos + {len(clips)} usable clips = about {unique:.0f}s of distinct picture "
                 f"for a {target_seconds:.0f}s video ({cov:.0%} coverage).")
    if cov < 0.6:
        notes.append("Coverage is low: shots will repeat noticeably. Add more of your own clips "
                     "(aim for 90s+ of footage and 8+ photos) or shorten the video.")
    return cov, notes


def build_compare_index(project: Project, cfg: dict, vision: bool = True) -> list[dict]:
    """Footage for a comparison: every product's own assets (ids P1-A01, P2-A01 ...) plus optional shared footage (X-A01)."""
    entries: list[dict] = []
    for i, (sub, label) in enumerate(zip(project.compare["products"], project.compare["labels"]), 1):
        log(f"Product {i} ({label}, {sub.asin}):")
        for e in build_index(sub, cfg, vision=vision):
            e = dict(e)
            e["id"], e["product"], e["label"] = f"P{i}-{e['id']}", str(i), label
            e["path"] = str(sub.assets_dir / e["file"])
            entries.append(e)
    if project.assets_dir.exists() and any(project.assets_dir.glob("*")):
        log("Shared footage (comparison folder):")
        for e in build_index(project, cfg, vision=vision):
            e = dict(e)
            e["id"], e["product"], e["label"] = f"X-{e['id']}", "all", "shared"
            e["path"] = str(project.assets_dir / e["file"])
            entries.append(e)
    return entries


def build_index_for(project: Project, cfg: dict, vision: bool = True) -> list[dict]:
    return build_compare_index(project, cfg, vision) if project.compare else build_index(project, cfg, vision)
