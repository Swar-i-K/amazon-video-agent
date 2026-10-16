"""Builds four throw-away products + a Top-4 list project for offline testing:
    python tests/make_list_fixture.py
    python -m adagent render LISTTEST01 --voice mock --no-vision --width 480 --height 270 --fps 15 --max-seconds 40
"""
import random
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(ROOT / "tests" / "make_compare_fixture.py")], check=True, capture_output=True)
P1 = ROOT / "projects" / "TEST000001"

for pid, brand, notes, link, perm in (("TEST000003", "ThirdCo", ["feels sturdy", "lid clicks shut", "fits a cup holder", "easy to rinse"], "test789", (1, 2, 0)),
                                      ("TEST000004", "FourthCo", ["slim body"], "test012", (2, 0, 1))):
    d = ROOT / "projects" / pid
    (d / "assets").mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(["01_front_white.jpg", "02_alt_color_white.jpg", "03_outdoor_scene.jpg", "04_portrait_tall.jpg", "08_scene_green.jpg"], 1):
        im = Image.open(P1 / "assets" / name).convert("RGB")
        ch = im.split()
        Image.merge("RGB", tuple(ch[j] for j in perm)).save(d / "assets" / f"p_{i:02d}.jpg", quality=92)
    (d / "facts.yaml").write_text(f"""asin: {pid}
product_name: {brand} Bottle
brand: {brand}
category: insulated water bottle
audience: commuters
affiliate_url: https://amzn.to/{link}
manufacturer_claims: [claim a]
specs: ["Capacity: 24 oz"]
hands_on_notes: {notes!r}
pros: [light]
cons: [no handle]
""".replace("'", '"'))

L = ROOT / "projects" / "LISTTEST01"
L.mkdir(parents=True, exist_ok=True)
(L / "list.yaml").write_text("""title: Top 4 insulated bottles for commuters
keywords: [insulated water bottle, best bottle for commuters, commuter bottle]
products:
  - id: TEST000001
    label: Best all-round
    best_for: daily commuting
  - id: TEST000002
    best_for: big capacity days
  - TEST000003
  - id: TEST000004
    label: Best for small bags
order: countdown
criteria: "Ranked by how well each suits a daily commute, using only the facts and notes I wrote."
category: insulated water bottle
audience: commuters
""")
rng = random.Random(5)
openers = ["Here's the thing about", "Let's talk about", "One detail worth noticing is", "What stands out first is",
           "Another thing you'll notice is", "It's worth pausing on", "Something to weigh up is"]
subjects = ["the lid and how it closes", "the overall shape in the hand", "how it sits in a cup holder", "the finish on the outside",
            "the opening for ice", "how easy it is to clean", "the weight when it is full", "the cap and its seal", "the size for a bag"]
tails = ["and that matters more than the spec sheet suggests when you carry it every single morning on the way to work.",
         "which is the kind of small thing you only appreciate after a week of using it around town.",
         "so it is worth thinking about how you plan to use it before you decide anything at all."]
plan = [("hook", "all", 2), ("criteria", "all", 2), ("item_4", "4", 7), ("item_3", "3", 7), ("item_2", "2", 7), ("item_1", "1", 7),
        ("recap", "all", 3), ("verdict", "all", 2)]
lines = ["# TITLE: Top 4 insulated bottles for commuters", "# INTRO: A calm countdown.", "# TAGS: bottle, list", "# ANGLE: list", ""]
n = 0
for sec, prod, count in plan:
    lines += ["", f"## {sec}"]
    for _ in range(count):
        n += 1
        ids = [f"P{prod}-A01", f"P{prod}-A03"] if prod != "all" else ["P1-A02", "P2-A02"]
        opts = [f"assets: {','.join(ids)}", "p: 2"]
        if prod != "all":
            opts.insert(0, f"product: {prod}")
        lines.append(f"[{n}] {rng.choice(openers)} {rng.choice(subjects)}, {rng.choice(tails)} | " + " | ".join(opts))
(L / "script.md").write_text("\n".join(lines) + "\n")
print("list fixture ready:", L)
