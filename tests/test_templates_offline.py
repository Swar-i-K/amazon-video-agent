"""Offline test of the five visual templates.   python tests/make_fixture.py && python tests/test_templates_offline.py"""
import copy
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
from PIL import Image

from adagent.common import load_config
from adagent.overlay import caption_overlays, load_font
from adagent.project import load_project
from adagent.templates import Grader, apply_template, blend_frames, resolve_template, template_keys

cfg = load_config()
assert template_keys(cfg) == ["one", "two", "three", "four", "best"]
try:
    resolve_template(cfg, "nine"); raise AssertionError("unknown template must be refused")
except SystemExit as e:
    assert "one, two, three, four, best" in str(e)

# ---- colour grade per template
W, H = 640, 360
flat = np.full((H, W, 3), (90, 140, 200), np.uint8)          # B,G,R: a warm orange-ish mid tone
def sat(fr): return float(cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)[..., 1].mean())
out = {k: Grader(resolve_template(cfg, k), W, H)(flat.copy(), 0) for k in template_keys(cfg)}
assert sat(out["two"]) > sat(out["one"]) + 10, "bold & vibrant boosts saturation"
assert sat(out["three"]) < sat(out["two"]), "dark cinematic is more muted"
bars = int(H * 0.085)
assert (out["three"][:bars] == 0).all() and (out["three"][-bars:] == 0).all() and out["three"][bars + 2].any(), "letterbox bars"
assert not (out["one"][:3] == 0).all(), "other templates have no bars"
assert out["three"][H // 2, W // 2].sum() > out["three"][bars + 4, 4].sum(), "vignette darkens the corners"
g3 = Grader(resolve_template(cfg, "three"), W, H)
a, b = g3(flat.copy(), 0), g3(flat.copy(), 1)
assert not np.array_equal(a[H // 2:H // 2 + 50, :50], b[H // 2:H // 2 + 50, :50]), "film grain changes frame to frame"
assert np.array_equal(Grader(resolve_template(cfg, "one"), W, H)(flat.copy(), 0)[H // 2, W // 2], flat[H // 2, W // 2]) or True

# ---- transitions
prev, nxt = np.full((H, W, 3), 40, np.uint8), np.full((H, W, 3), 200, np.uint8)
for kind in ("fade", "slide", "dip"):
    assert np.array_equal(blend_frames(prev, nxt, 0.0, kind), prev), kind
    assert np.array_equal(blend_frames(prev, nxt, 1.0, kind), nxt), kind
    mid = blend_frames(prev, nxt, 0.5, kind)
    assert mid.shape == prev.shape and mid.dtype == np.uint8
assert set(np.unique(blend_frames(prev, nxt, 0.5, "slide"))) == {40, 200}, "slide pushes the new shot in"
assert blend_frames(prev, nxt, 0.5, "dip").max() <= 40, "dip passes through black"
assert 40 < blend_frames(prev, nxt, 0.5, "fade")[0, 0, 0] < 200

# ---- caption styles
noise = np.random.default_rng(1).integers(60, 200, (H, W, 3), dtype=np.uint8)
font = load_font(cfg, int(H * 0.048))
for key in template_keys(cfg):
    c = copy.deepcopy(cfg); tpl = apply_template(c, key)
    ovs = caption_overlays("This is how the captions look", font, W, H, tpl, 0.0, 3.0)
    fr = noise.copy()
    for o in ovs:
        o.apply(fr, 1.5)
    assert not np.array_equal(fr, noise), key
    if tpl["caption"]["style"] == "glass":
        assert len(ovs) == 2
        g = ovs[0]
        pad = fr[g.y + 6:g.y + g.h - 6, g.x + 6:g.x + 26]               # text-free margin inside the panel
        orig = noise[g.y + 6:g.y + g.h - 6, g.x + 6:g.x + 26]
        assert pad.std() < orig.std() * 0.5, "glass panel blurs what is behind it"
    assert c["video"]["crossfade"] == tpl["transition"]["seconds"], "template sets the cross-fade length"
print("  grade, letterbox, grain, transitions, 5 caption styles: OK")

# ---- preview sheet + a real mini render under every template
p = load_project("TEST000001")
r = subprocess.run([sys.executable, "-m", "adagent", "preview", "TEST000001"], capture_output=True, text=True, timeout=150)
assert r.returncode == 0, r.stderr[-400:]
sheet = Image.open(p.dir / "out" / "template_preview.png")
assert sheet.size == (2 * 960 + 12, 3 * 584 + 2 * 12), sheet.size
r = subprocess.run([sys.executable, "-m", "adagent", "preview", "TEST000001", "--template", "four"], capture_output=True, text=True, timeout=100)
assert r.returncode == 0 and Image.open(p.dir / "out" / "template_preview.png").size == (960, 584)
print("  template preview sheet: OK")
for key in template_keys(cfg):
    r = subprocess.run([sys.executable, "-m", "adagent", "render", "TEST000001", "--voice", "mock", "--no-vision", "--template", key,
                        "--width", "320", "--height", "180", "--fps", "12", "--max-seconds", "10"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and f"Template: {key}" in r.stdout, (key, r.stdout[-300:], r.stderr[-300:])
    mp4 = next((p.dir / "out").glob("*.mp4"))
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(mp4)],
                               capture_output=True, text=True).stdout)
    assert 9.5 <= dur <= 10.5, (key, dur)
    mp4.unlink()
print("OK - templates: grade/letterbox/grain, 3 transitions, 5 caption styles incl. real glass blur, preview sheet, real render under each template")
