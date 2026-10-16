"""Template preview: one sample picture per template (caption, label chip, rank badge, callout) on one of your own photos."""
from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .assets import load_image_bgr
from .cinematic import CinematicProduct
from .common import log
from .overlay import build_overlays, load_font
from .render import compose
from .templates import Grader, apply_template, template_keys, tpl_of
from .timeline import Row


def _photo(project, entries):
    imgs = [e for e in entries if e["usable"] and e["kind"] == "image"]
    if imgs:
        e = imgs[0]
        path = Path(e["path"]) if e.get("path") else project.assets_dir / e["file"]
        return load_image_bgr(path)
    log("  (no photo found; using a plain sample picture)")
    img = np.full((900, 1200, 3), 245, np.uint8)
    cv2.rectangle(img, (480, 180), (720, 760), (160, 90, 30), -1)
    return img


def render_template_preview(project, cfg, entries, only: str | None = None) -> Path:
    keys = [only] if only else template_keys(cfg)
    W, H = 960, 540
    src = _photo(project, entries)
    tiles = []
    for key in keys:
        c = copy.deepcopy(cfg)
        c["video"]["width"], c["video"]["height"] = W, H
        c["disclosure"]["mode"] = "description_only"            # keep the sample clean
        tpl = apply_template(c, key)
        try:
            base = CinematicProduct(src, W, H, variant=0, seed=1, palette=tpl["backdrop"]).render(2.6, 5.0)
        except ValueError:                                       # photo has no clean backdrop: show it the normal way
            base = compose(src, W, H)
        rows = [Row(0.0, 6.0, 0.4, {"id": "1", "section": "item_1", "product": "1", "onscreen": "4 towels in a pack",
                                    "text": "This is how the captions look in this template.", "asset_ids": []})]
        ovs = build_overlays(rows, 10.0, c, W, H, "Preview", [], product_labels=["Sample product"], kind="list",
                             rank_notes=["Best for hot days"])
        fr = Grader(tpl_of(c), W, H)(np.ascontiguousarray(base), 0)
        for o in ovs:
            o.apply(fr, 2.0)
        rgb = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        bar = Image.new("RGB", (W, 44), (18, 18, 22))
        ImageDraw.Draw(bar).text((14, 11), f"{key}  -  {tpl.get('label', '')}   |   caption: {tpl['caption']['style']},  "
                                           f"transition: {tpl['transition']['kind']}", font=load_font(c, 20), fill=(235, 235, 235))
        tile = Image.new("RGB", (W, H + 44))
        tile.paste(bar, (0, 0))
        tile.paste(rgb, (0, 44))
        tiles.append(tile)
    cols = 1 if len(tiles) == 1 else 2
    rows_n = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * W + (cols - 1) * 12, rows_n * (H + 44) + (rows_n - 1) * 12), (8, 8, 10))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * (W + 12), (i // cols) * (H + 44 + 12)))
    out = project.out_dir / "template_preview.png"
    sheet.save(out)
    return out
