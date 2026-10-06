"""Builds a throw-away test project (synthetic photos/clips + a long script) so the pipeline can be
exercised end-to-end with no network and no API keys:

    python tests/make_fixture.py
    python -m adagent render TEST000001 --voice mock --no-vision --width 640 --height 360 --fps 24
"""
import math
import random
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "projects" / "TEST000001"
A = P / "assets"
A.mkdir(parents=True, exist_ok=True)


def bottle(w, h, bg=(255, 255, 255), color=(30, 90, 160), alpha=False):
    im = Image.new("RGBA" if alpha else "RGB", (w, h), (0, 0, 0, 0) if alpha else bg)
    d = ImageDraw.Draw(im)
    bw, bh = int(w * 0.28), int(h * 0.74)
    x0, y0 = (w - bw) // 2, int(h * 0.16)
    d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=int(bw * 0.25), fill=color)
    d.rounded_rectangle([x0 + bw * 0.2, y0 - h * 0.07, x0 + bw * 0.8, y0 + 4], radius=12, fill=(40, 40, 40))
    d.rectangle([x0 + bw * 0.15, y0 + bh * 0.4, x0 + bw * 0.85, y0 + bh * 0.55], fill=(240, 240, 240))
    return im


def scene(w, h, c1, c2):
    im = Image.new("RGB", (w, h))
    px = im.load()
    for y in range(h):
        t = y / h
        for x in range(0, w, 1):
            px[x, y] = tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([w * 0.42, h * 0.2, w * 0.58, h * 0.9], radius=40, fill=(30, 90, 160))
    return im


bottle(1600, 1600).save(A / "01_front_white.jpg", quality=92)
bottle(1600, 1600, color=(160, 60, 40)).save(A / "02_alt_color_white.jpg", quality=92)
scene(1920, 1080, (250, 200, 120), (60, 90, 140)).save(A / "03_outdoor_scene.jpg", quality=92)
bottle(1080, 1620, bg=(235, 238, 242)).save(A / "04_portrait_tall.jpg", quality=92)
bottle(640, 640, color=(20, 120, 90)).save(A / "05_small_lowres.jpg", quality=90)
bottle(1400, 1400, alpha=True).save(A / "06_transparent.png")
bottle(1500, 1000, bg=(255, 255, 255), color=(90, 60, 140)).save(A / "07_landscape_white.webp", quality=90)
scene(2400, 1350, (90, 150, 110), (20, 40, 60)).save(A / "08_scene_green.jpg", quality=92)


def clip(name, w, h, fps, secs, src="testsrc2"):
    out = A / name
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"{src}=size={w}x{h}:rate={fps}", "-t", str(secs), "-c:v", "libx264", "-pix_fmt", "yuv420p", out], check=True)


clip("clip_a_landscape.mp4", 1920, 1080, 30, 10)
clip("clip_b_portrait.mp4", 1080, 1920, 24, 7, src="testsrc")

(P / "facts.yaml").write_text("""asin: TEST000001
product_name: Test Bottle 20 oz
brand: TestCo
category: insulated water bottle
audience: commuters
affiliate_url: https://amzn.to/test123
manufacturer_claims: [claim one, claim two]
specs: ["Capacity: 20 oz"]
hands_on_notes: [fits a cup holder, lid is easy to open, cap feels solid, wide mouth takes ice cubes]
pros: [light, easy to clean]
cons: [no handle]
""")

rng = random.Random(7)
openers = ["Here's the thing about", "Let's talk about", "One detail worth noticing is", "If you carry a bottle every day,",
           "What stands out first is", "Another thing you'll notice is", "It's worth pausing on"]
subjects = ["the lid and how it closes", "the overall shape in the hand", "how it sits in a cup holder", "the finish on the outside",
            "the wide opening for ice", "how easy it is to clean", "the weight when it is full", "the cap and its seal",
            "the base and how stable it feels", "the color options", "the size for a work bag", "the strap loop"]
tails = ["and that matters more than the spec sheet suggests when you use it every single morning on the way to work.",
         "which is the kind of small thing you only appreciate after a week of carrying it around town.",
         "so it is worth thinking about how you actually plan to use it before you decide anything at all.",
         "and honestly that is where a lot of bottles in this category tend to differ from one another."]
secs = ["hook"] * 2 + ["context"] * 4 + ["overview"] * 4 + ["features"] * 13 + ["in_use"] * 5 + ["limits"] * 3 + ["who_for"] * 2 + ["who_skip"] * 2 + ["verdict"] * 2
lines = ["# TITLE: Test Bottle 20 oz honest look", "# INTRO: A calm look at a test product.", "# TAGS: test, bottle", "# ANGLE: first_impressions", ""]
cur = None
ids = [f"A0{i}" for i in range(1, 9)] + ["A09", "A10"]
for i, s in enumerate(secs, 1):
    if s != cur:
        cur = s
        lines += ["", f"## {s}"]
    text = f"{rng.choice(openers)} {rng.choice(subjects)}, {rng.choice(tails)}"
    opts = f"assets: {','.join(rng.sample(ids, 2))} | p: {rng.choice([1, 2, 3])}"
    if i % 9 == 0:
        opts = "callout: 20 oz capacity | " + opts
    lines.append(f"[{i}] {text} | {opts}")
(P / "script.md").write_text("\n".join(lines) + "\n")
print("fixture ready:", P)
