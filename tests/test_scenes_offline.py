"""Offline test of scene planning + AI-scene labelling.   python tests/test_scenes_offline.py  (run make_fixture.py first)"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adagent import assets as A
from adagent import cli
from adagent import scenes as SC
from adagent.common import load_config
from adagent.project import load_project

cfg = load_config()
cfg["scenes"]["max_product_scenes"] = 2          # the shipped default is higher; this test checks the cap itself
p = load_project("TEST000001")

# --- prompt rules
assert SC.check_prompt("A smiling woman looks at the camera")                  # face
assert SC.check_prompt("Frost spreads across the towel instantly")             # product effect
assert SC.check_prompt("A box with the brand logo and a $19 price")            # brand/price/text
assert not SC.check_prompt("Hands from behind drape a folded cloth over a gym bag in warm morning light, slow push-in")

good = "Runner seen from behind jogging along a sunlit park path at golden hour, long shadows, slow push-in, calm mood"
reply = {"scenes": [
    {"id": "s01", "type": "mood", "ref": "none", "prompt": good},
    {"id": "s02", "type": "mood", "ref": "none", "prompt": "Close-up of a smiling face in the sun"},          # dropped
    {"id": "s03", "type": "mood", "ref": "none", "prompt": "Ice crystals form on the fabric, steam rising"},  # dropped
    {"id": "s04", "type": "product", "ref": "A01", "prompt": "Product on a pale table, slow camera push-in, soft window light"},
    {"id": "s05", "type": "product", "ref": "A02", "prompt": "Product rotating gently on a pale table, soft light, slow drift"},
    {"id": "s06", "type": "product", "ref": "A03", "prompt": "Product on a pale table, slow tilt, soft light"},      # 3rd product -> mood
    {"id": "s07", "type": "product", "ref": "A99", "prompt": "Product on a shelf, slow push-in, gentle light"},    # bad ref -> mood
    {"id": "s08", "type": "mood", "ref": "none", "prompt": "Gym bag open on a bench, towel corner visible, slow drift, soft light"},
]}
SC.ask = lambda *a, **k: json.dumps(reply)
entries = A.build_index(p, cfg, vision=False)
scenes = SC.generate_scenes(p, cfg, entries, n=8)
assert len(scenes) == 6, [s["id"] for s in scenes]
assert sum(1 for s in scenes if s["type"] == "product") == 2
assert all(s["ref"] == "none" for s in scenes if s["type"] == "mood")
assert [s["id"] for s in scenes] == [f"s0{i}" for i in range(1, 7)]

md = SC.scenes_to_md(scenes, p)
back = SC.md_to_scenes(md)
assert back == scenes, "markdown round-trip must be lossless"

# --- CLI command writes the file and refuses to overwrite without --force
(Path(p.dir) / "scenes.md").unlink(missing_ok=True)
cli.main(["scenes", "TEST000001", "--no-vision"])
assert (p.dir / "scenes.md").exists()
try:
    cli.main(["scenes", "TEST000001", "--no-vision"])
    raise AssertionError("should refuse to overwrite")
except SystemExit as e:
    assert "already exists" in str(e)

# --- ai_* clips are allowed even when 'shows_product' is false (mood b-roll), other clips are not
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=24", "-t", "6",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p.assets_dir / "ai_s01.mp4")], check=True)
A.describe = lambda cfg, frames, kind: {"description": "sunlit park path", "shows_product": False, "has_human_face": False,
                                        "has_price_or_rating_ui": False, "has_third_party_logo_or_watermark": False}
(p.cache_dir / "assets_index.json").unlink(missing_ok=True)
ents = A.build_index(p, cfg, vision=True)
by = {e["file"]: e for e in ents}
assert by["ai_s01.mp4"]["usable"], by["ai_s01.mp4"]
assert not by["clip_a_landscape.mp4"]["usable"] and "product" in by["clip_a_landscape.mp4"]["reason"]
print("OK - scene planning: unsafe prompts dropped, product scenes capped at 2, md round-trip, CLI, ai_* clips allowed")
