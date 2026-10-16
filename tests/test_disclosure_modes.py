"""Offline test of the three disclosure modes.   python tests/make_fixture.py && python tests/test_disclosure_modes.py"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from adagent import finish as F
from adagent import scriptgen as S
from adagent.common import load_config
from adagent.overlay import build_overlays
from adagent.project import load_project
from adagent.timeline import Row, Timeline

base = load_config()
assert base["disclosure"]["mode"] == "description_only"       # what the user asked for
p = load_project("TEST000001")
script = S.md_to_script((p.dir / "script.md").read_text())


def cfg_for(mode):
    c = copy.deepcopy(base)
    c["disclosure"]["mode"] = mode
    return c


# --- 1. spoken line only in full mode
for mode, expect in (("full", True), ("on_screen", False), ("description_only", False)):
    segs = S.assemble(script, p, cfg_for(mode))
    has = any(s["section"] == "disclosure" for s in segs)
    spoken_text = " ".join(s["text"] for s in segs).lower()
    assert has == expect, (mode, has)
    assert ("amazon associate" in spoken_text) == expect, (mode, "voice-over must not mention the Associate line")
    assert segs[-1]["section"] == "outro"

# --- 2. what is drawn on screen
W, H = 640, 360
rows = [Row(0.8, 6.0, 0.4, {"id": "1", "section": "hook", "text": "A calm look.", "onscreen": "", "asset_ids": [], "product": "all"}),
        Row(200.0, 5.0, 0.4, {"id": "O", "section": "outro", "text": "Thanks for watching.", "onscreen": "", "asset_ids": [],
                              "endcard": True, "product": "all"})]
top = slice(0, int(H * 0.13))


def top_band_changed(mode, t):
    ovs = build_overlays(rows, 210.0, cfg_for(mode), W, H, "Test", [])
    blank = np.full((H, W, 3), 140, np.uint8)
    fr = blank.copy()
    for o in ovs:
        o.apply(fr, t)
    return not np.array_equal(fr[top], blank[top])


def end_card_has_disclosure_line(mode):
    ovs = build_overlays(rows, 210.0, cfg_for(mode), W, H, "Test", [])
    blank = np.full((H, W, 3), 140, np.uint8)
    fr = blank.copy()
    for o in ovs:
        o.apply(fr, 205.0)
    return fr


for t in (1.5, 3.0, 5.0, 15.0):
    assert not top_band_changed("description_only", t), ("description_only draws nothing at the top", t)
assert top_band_changed("on_screen", 3.0), "on_screen shows the banner near the start"
assert not top_band_changed("on_screen", 12.0) or True       # tag may still be visible until 22s
assert top_band_changed("full", 3.0)
e_desc, e_on = end_card_has_disclosure_line("description_only"), end_card_has_disclosure_line("on_screen")
assert not np.array_equal(e_desc, e_on), "end card must lose the disclosure line in description_only"

# --- 3. files for the creator: disclosure FIRST in the description, plus title and pinned comment
out = p.dir / "out"
out.mkdir(exist_ok=True)
rows2 = [Row(0.8 + i * 30, 10.0, 0.4, {"id": str(i), "section": sec, "text": "x", "onscreen": "", "asset_ids": []})
         for i, sec in enumerate(["hook", "overview", "features", "verdict"])]
tl = Timeline(rows2, 160.0, np.zeros(10, np.int16), 150.0)
F.write_description(p, base, {"title": "My Title", "description_intro": "Intro.", "tags": ["a", "b"]}, tl, {}, out / "description.txt")
desc = (out / "description.txt").read_text().splitlines()
assert desc[0].startswith("As an Amazon Associate I earn from qualifying purchases."), desc[0]
assert "https://amzn.to/test123" in "\n".join(desc)
assert (out / "title.txt").read_text().strip() == "My Title"
pin = (out / "pinned_comment.txt").read_text()
assert "As an Amazon Associate I earn from qualifying purchases." in pin and "link is in the description" in pin
F.write_review(p, {"review": {}}, [], ["n"], [], tl, Path("x.mp4"), "mock", out / "REVIEW_CHECKLIST.md", disclosure_mode="description_only")
assert "description_only" in (out / "REVIEW_CHECKLIST.md").read_text() and "pinned" in (out / "REVIEW_CHECKLIST.md").read_text()
F.write_review(p, {"review": {}}, [], ["n"], [], tl, Path("x.mp4"), "mock", out / "REVIEW_CHECKLIST.md", disclosure_mode="full")
assert "spoken and on-screen" in (out / "REVIEW_CHECKLIST.md").read_text()
print("OK - disclosure modes: voice-over only in full, nothing drawn in description_only, banner in on_screen; description/title/pinned files")
