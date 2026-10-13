"""Builds two throw-away products + a comparison project so the comparison feature can be tested offline:
    python tests/make_compare_fixture.py
    python -m adagent render CMPTEST001 --voice mock --no-vision --width 640 --height 360 --fps 24 --max-seconds 50
"""
import random
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(ROOT / "tests" / "make_fixture.py")], check=True, capture_output=True)

# ---- product 2: recoloured copies of product 1's photos, a different brand and link
P1, P2 = ROOT / "projects" / "TEST000001", ROOT / "projects" / "TEST000002"
(P2 / "assets").mkdir(parents=True, exist_ok=True)
for i, name in enumerate(["01_front_white.jpg", "02_alt_color_white.jpg", "03_outdoor_scene.jpg", "04_portrait_tall.jpg",
                          "07_landscape_white.webp", "08_scene_green.jpg"], 1):
    im = Image.open(P1 / "assets" / name).convert("RGB")
    r, g, b = im.split()
    Image.merge("RGB", (b, r, g)).save(P2 / "assets" / f"other_{i:02d}.jpg", quality=92)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30", "-t", "9",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(P2 / "assets" / "other_clip.mp4")], check=True)
(P2 / "facts.yaml").write_text("""asin: TEST000002
product_name: Other Bottle 32 oz
brand: OtherCo
category: insulated water bottle
audience: commuters
affiliate_url: https://amzn.to/test456
manufacturer_claims: [claim a, claim b]
specs: ["Capacity: 32 oz"]
hands_on_notes: [heavier than expected, does not fit a cup holder]
pros: [large capacity]
cons: [bulky]
""")

# ---- the comparison project
C = ROOT / "projects" / "CMPTEST001"
C.mkdir(parents=True, exist_ok=True)
(C / "compare.yaml").write_text("""title: TestCo vs OtherCo insulated bottles
products: [TEST000001, TEST000002]
category: insulated water bottle
audience: commuters
compare_points: [capacity, cup holder fit]
""")
rng = random.Random(3)
openers = ["Here's the thing about", "Let's talk about", "One detail worth noticing is", "What stands out first is",
           "Another thing you'll notice is", "It's worth pausing on", "Something to weigh up is"]
subjects = ["the lid and how it closes", "the overall shape in the hand", "how it sits in a cup holder", "the finish on the outside",
            "the opening for ice", "how easy it is to clean", "the weight when it is full", "the cap and its seal", "the size for a bag"]
tails = ["and that matters more than the spec sheet suggests when you carry it every single morning on the way to work.",
         "which is the kind of small thing you only appreciate after a week of using it around town.",
         "so it is worth thinking about how you plan to use it before you decide anything at all."]
plan = [("hook", "all", 2), ("context", "all", 3), ("overview", "all", 2), ("product_1", "1", 8), ("product_2", "2", 8),
        ("head_to_head", "all", 8), ("who_for", "all", 3), ("verdict", "all", 3)]
lines = ["# TITLE: TestCo vs OtherCo bottles compared", "# INTRO: A calm side by side look.", "# TAGS: bottle, comparison",
         "# ANGLE: comparison", ""]
n = 0
for sec, prod, count in plan:
    lines += ["", f"## {sec}"]
    for _ in range(count):
        n += 1
        ids = ["P1-A01", "P1-A03", "P1-A09"] if prod == "1" else ["P2-A01", "P2-A03", "P2-A07"] if prod == "2" else ["P1-A02", "P2-A02"]
        opts = [f"assets: {','.join(rng.sample(ids, 2))}", "p: 2"]
        if prod != "all":
            opts.insert(0, f"product: {prod}")
        lines.append(f"[{n}] {rng.choice(openers)} {rng.choice(subjects)}, {rng.choice(tails)} | " + " | ".join(opts))
(C / "script.md").write_text("\n".join(lines) + "\n")
print("comparison fixture ready:", C)
