"""Choose the footage for ONE video out of everything uploaded to a product's assets/ folder.

Free and local (uses the pre-scan, no AI). The pick is: good quality, no near-duplicates, varied (looks/colours/angles differ), and fresh
(footage already used in earlier videos of the same product is avoided, so several videos of one product do not look alike). Only the picked
files are then AI-checked, which keeps Groq usage small however many files you upload.
The pick is saved in selection.json so 'script' and 'render' use exactly the same files; run 'script' again for a new pick.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from .prescan import hamming

USAGE_KEEP = 30           # remember the last N videos per product


# ----------------------------- budgets ---------------------------------------
def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def budgets(cfg: dict, kind: str | None, n_products: int = 1) -> tuple[int, int]:
    """(images, clips) to pick per product for this video. 'auto' sizes it from the video length."""
    a = cfg.get("assets", {})
    shots = cfg["length"]["target_minutes"] * 60.0 / max(2.0, float(cfg["video"].get("shot_seconds", 4.5)))
    if kind in ("compare", "list"):
        per = max(1, n_products)
        auto_i = _clamp(round(shots * 0.55 / per), 5, 14)
        auto_c = _clamp(round(shots * 0.22 / per), 2, 5)
        i, c = a.get("per_product_images", "auto"), a.get("per_product_clips", "auto")
    else:
        auto_i, auto_c = _clamp(round(shots * 0.4), 8, 36), _clamp(round(shots * 0.2), 3, 12)
        i, c = a.get("images_per_video", "auto"), a.get("clips_per_video", "auto")
    return (auto_i if i == "auto" else int(i)), (auto_c if c == "auto" else int(c))


# ----------------------------- usage memory ----------------------------------
def _usage_path(pdir: Path) -> Path:
    return pdir / "asset_usage.json"


def load_usage(pdir: Path) -> dict:
    p = _usage_path(pdir)
    try:
        return json.loads(p.read_text()) if p.exists() else {"videos": []}
    except Exception:
        return {"videos": []}


def uses_of(pdir: Path) -> tuple[Counter, dict]:
    """(how many earlier videos used each file, {clip file: start times already used})."""
    uses: Counter = Counter()
    starts: dict = defaultdict(list)
    for v in load_usage(pdir).get("videos", [])[-USAGE_KEEP:]:
        for name, info in (v.get("files") or {}).items():
            uses[name] += 1
            starts[name] += list(info.get("starts") or [])
    return uses, starts


def record_usage(project, shots: list, angle: str | None, title: str = "") -> list[str]:
    """After a full render: remember which files (and which clip moments) the video used. Returns the folders written."""
    per: dict = defaultdict(lambda: defaultdict(lambda: {"shots": 0, "starts": []}))
    for s in shots:
        e = s.asset
        pdir = project.dir
        if project.compare and str(e.get("product", "")).isdigit():
            pdir = project.compare["products"][int(e["product"]) - 1].dir
        d = per[pdir][e["file"]]
        d["shots"] += 1
        if e.get("kind") == "clip":
            d["starts"].append(round(float(getattr(s, "clip_in", 0.0)), 1))
    key = hashlib.sha1(f"{project.asin}|{title}|{angle}".encode()).hexdigest()[:10]
    written = []
    for pdir, files in per.items():
        data = load_usage(pdir)
        vids = [v for v in data.get("videos", []) if v.get("key") != key]      # re-rendering the same video replaces its entry
        vids.append({"key": key, "date": date.today().isoformat(), "project": project.asin, "angle": angle,
                     "files": {k: {"shots": v["shots"], "starts": sorted(set(v["starts"]))} for k, v in files.items()}})
        data["videos"] = vids[-USAGE_KEEP:]
        _usage_path(pdir).write_text(json.dumps(data, indent=1))
        written.append(str(pdir))
    return written


def ordered_windows(feat: dict, used_starts: list[float]) -> list[float]:
    """Best moments of a clip first; moments an earlier video already used go last."""
    wins = sorted(feat.get("windows") or [], key=lambda w: -w["score"])
    fresh = [w["start"] for w in wins if all(abs(w["start"] - u) >= 2.0 for u in used_starts)]
    old = [w["start"] for w in wins if w["start"] not in fresh]
    return fresh + old


# ----------------------------- saved choices ---------------------------------
class SelectionStore:
    """projects/<video project>/selection.json: {"products": {"<ASIN or X>": [files]}}. Plain text: you may edit the list yourself."""

    def __init__(self, path: Path, write: bool):
        self.path, self.write, self.dirty = path, write, False
        try:
            self.data = json.loads(path.read_text()) if path.exists() else {}
        except Exception:
            self.data = {}

    def get(self, key: str):
        return (self.data.get("products") or {}).get(key)

    def put(self, key: str, files: list[str]) -> None:
        self.data.setdefault("products", {})[key] = sorted(files)
        self.dirty = True

    def save(self) -> None:
        if self.write and self.dirty:
            self.data["created"] = date.today().isoformat()
            self.path.write_text(json.dumps(self.data, indent=1))


# ----------------------------- choosing ---------------------------------------
def _dist(a: dict, b: dict) -> float:
    """0..1: how different two files look (colour fingerprint + shape fingerprint)."""
    colour = sum(abs(x - y) for x, y in zip(a["hist"], b["hist"])) / 4.0
    shape = hamming(int(a["dhash"], 16), int(b["dhash"], 16)) / 64.0
    return min(1.0, 0.6 * colour + 0.4 * min(1.0, shape * 2))


def _gate(name: str, f: dict, cfg: dict, allow: set) -> str | None:
    """Free quality gate. Returns the reason a file cannot be used, or None."""
    a = cfg.get("assets", {})
    if f.get("unreadable"):
        return "could not be read"
    if name.lower() in allow:
        return None
    if f["kind"] == "image":
        if min(f["w"], f["h"]) < int(a.get("min_image_px", 480)):
            return f"too small ({f['w']}x{f['h']})"
        if f["std"] < 4:
            return "almost blank"
        if f["sharp"] < float(a.get("min_sharpness", 12)):
            return "too blurry"
    else:
        if f["dur"] < 1.5:
            return "clip shorter than 1.5 s"
        if f["sharp"] < float(a.get("min_sharpness", 12)) * 0.5:
            return "too blurry"
    return None


def choose(files: dict[str, dict], cfg: dict, n_img: int, n_clip: int, uses: Counter, pins: set, allow: set
           ) -> tuple[list[str], dict[str, str]]:
    """Returns (chosen file names, {skipped file: reason}). Fewer files than the budget: all good ones are used."""
    a = cfg.get("assets", {})
    penalty = float(a.get("reuse_penalty", 0.5))
    skipped: dict[str, str] = {}
    pool: dict[str, dict] = {}
    for name, f in files.items():
        why = _gate(name, f, cfg, allow)
        if why and name not in pins:
            skipped[name] = why
        else:
            pool[name] = f
    chosen: list[str] = []
    # pinned files and the AI ad clips (ai_*.mp4 made from your photos) always go in (AI clips up to max_ai_clips)
    ai = sorted((n for n in pool if n.lower().startswith("ai_")), key=lambda n: -pool[n]["quality"])[: int(a.get("max_ai_clips", 8))]
    for n in sorted(pins & set(pool)) + ai:
        if n not in chosen:
            chosen.append(n)
    for n in pool:
        if n.lower().startswith("ai_") and n not in chosen:
            skipped[n] = "more AI clips than max_ai_clips"
    dup_radius = int(a.get("duplicate_distance", 5))
    for kind, budget in (("image", n_img), ("clip", n_clip)):
        cands = sorted((n for n, f in pool.items() if f["kind"] == kind and n not in chosen and not n.lower().startswith("ai_")),
                       key=lambda n: -pool[n]["quality"])
        kept: list[str] = []
        for n in cands:      # near-duplicates: keep the best of each look-alike group
            twin = next((k for k in kept + [c for c in chosen if pool[c]["kind"] == kind]
                         if hamming(int(pool[n]["dhash"], 16), int(pool[k]["dhash"], 16)) <= dup_radius
                         and _dist(pool[n], pool[k]) < 0.12), None)
            if twin:
                skipped[n] = f"near-duplicate of {twin}"
            else:
                kept.append(n)
        have = [c for c in chosen if pool[c]["kind"] == kind]
        room = max(0, budget - len(have))
        picked: list[str] = []

        def base(n: str) -> float:
            u = uses[n]
            return pool[n]["quality"] + (0.4 if u == 0 else -penalty * min(u, 3))

        while kept and len(picked) < room:      # greedy: quality + freshness, spread over different looks
            ref = have + picked
            best = max(kept, key=lambda n: 0.6 * base(n) + 0.4 * min((_dist(pool[n], pool[r]) for r in ref), default=1.0))
            picked.append(best)
            kept.remove(best)
        for n in kept:
            skipped[n] = "not needed for this video (kept for another one)"
        chosen += picked
    return chosen, skipped


def write_report(path: Path, project_name: str, rows: list) -> None:
    """rows: (label, chosen, skipped, files) per product; a plain-language list of what was picked and why the rest was not."""
    L = [f"# Footage selection for {project_name}", "",
         "Picked automatically from everything in assets/ (free local scan, no AI). Run the script step again for a different pick.",
         "To force a file in, list it under  assets_pin:  in facts.yaml. To change the saved pick by hand, edit selection.json.", ""]
    for label, chosen, skipped, files in rows:
        L.append(f"## {label}: {len(chosen)} picked, {len(skipped)} not used")
        for n in chosen:
            f = files.get(n, {})
            L.append(f"- PICKED  {n}  ({f.get('kind', '?')}, quality {f.get('quality', 0):.2f})")
        for n, why in sorted(skipped.items()):
            L.append(f"- skipped {n}: {why}")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")
