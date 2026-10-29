"""Offline test of Top-N list videos.   python tests/make_list_fixture.py && python tests/test_list_offline.py"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from adagent import assets as A
from adagent import cli
from adagent import finish as F
from adagent import render as R
from adagent import scriptgen as S
from adagent import timeline as T
from adagent import tts as V
from adagent.common import load_config
from adagent.overlay import build_overlays
from adagent.project import load_project, validate_facts

cfg = load_config()
p = load_project("LISTTEST01")
c = p.compare
assert c["kind"] == "list" and c["labels"] == ["TestCo", "OtherCo", "ThirdCo", "FourthCo"], c["labels"]
assert c["ranks"] == [1, 2, 3, 4] and c["order"] == "countdown" and "commute" in c["criteria"]
assert c["rank_labels"] == ["Best all-round", "", "", "Best for small bags"] and c["best_for"][0] == "daily commuting"
assert p.facts["keywords"][0] == "insulated water bottle" and p.facts["angle"] == "list"
errs, warns = validate_facts(p)
assert not errs, errs
assert any("YOUR decision" in w for w in warns) and any("[TEST000004] only 1 hands_on_notes" in w for w in warns), warns

# ---- list size limits and a missing criteria warning
lf = p.dir / "list.yaml"
orig = lf.read_text()
lf.write_text(re.sub(r"products:.*?order:", "products: [TEST000001, TEST000002]\norder:", orig, flags=re.S))
try:
    load_project("LISTTEST01"); raise AssertionError("a list needs at least 3 products")
except SystemExit as e:
    assert "3 to 9" in str(e)
lf.write_text(orig.replace("order: countdown", "order: sideways"))
try:
    load_project("LISTTEST01"); raise AssertionError("bad order")
except SystemExit as e:
    assert "countdown" in str(e)
lf.write_text(re.sub(r"criteria:.*\n", "", orig))
_, w2 = validate_facts(load_project("LISTTEST01"))
assert any("criteria" in w for w in w2)
lf.write_text(orig)
(p.dir / "compare.yaml").write_text("title: x\nproducts: [TEST000001, TEST000002]\ncategory: c\n")
try:
    load_project("LISTTEST01"); raise AssertionError("compare.yaml + list.yaml together must be refused")
except SystemExit as e:
    assert "both" in str(e)
(p.dir / "compare.yaml").unlink()

# ---- facts block + prompts
fy = S.facts_yaml(p)
assert "LIST: Top 4" in fy and "ORDER: countdown (present the sections in this order: item_4, item_3, item_2, item_1)" in fy
assert "rank #1; label: TestCo" in fy and "rank_label: Best all-round" in fy and "KEYWORDS: insulated water bottle" in fy
assert S.pick_angle(p, cfg, None) == "list"

seen = {"gen": 0, "expand": 0, "check": None}


def fake_ask(model, system, user, **kw):
    if system == S.FACTCHECK_SYSTEM:
        seen["check"] = user
        return json.dumps({"issues": []})
    assert system == S.LIST_SYSTEM, "list projects must use the list writer"
    if "CURRENT SCRIPT" in user:
        seen["expand"] += 1
        assert '"product": "1-9|all"' in user
        words = int(re.search(r"about (\d+) words of NEW", user).group(1))
        new = [{"after_id": "5", "section": "item_4", "product": "4", "text": f"Extra detail {seen['expand']}-{i} about the fourth pick in plain spoken words.",
                "onscreen": "", "asset_ids": ["P4-A01"], "source": "product", "priority": 3} for i in range(max(4, words // 24))]
        return json.dumps({"segments": new})
    seen["gen"] += 1
    assert S.LIST_GUIDE in user and "PRODUCT 4" in user and "item_3" in user and "P4-A01" in user
    words = int(re.search(r"about (\d+) words", user).group(1))
    n = max(14, int(words * 0.30) // 22)
    secs = ["hook", "criteria", "item_4", "item_3", "item_2", "item_1", "recap", "verdict"]
    segs = []
    for i in range(1, n + 1):
        sec = secs[min(len(secs) - 1, (i * len(secs)) // (n + 1))]
        prod = sec[5:] if sec.startswith("item_") else "all"
        segs.append({"id": i, "section": sec, "product": prod, "source": "product", "priority": 2, "onscreen": "",
                     "asset_ids": [f"P{prod}-A01"] if prod != "all" else [],
                     "text": f"Line number {i} written in plain spoken words about this part of the list, nothing more to add."})
    return json.dumps({"title": "Top 4 insulated bottles", "description_intro": "x", "tags": ["bottle", "Commuter Bottle"], "segments": segs})


S.ask = fake_ask
key = cli.choose_length(cfg, p, None)
assert key == "list" and (cfg["length"]["min_minutes"], cfg["length"]["max_minutes"], cfg["length"]["target_minutes"]) == (5.5, 8.8, 7.5)
entries = A.build_index_for(p, cfg, vision=False)
assert {e["product"] for e in entries} == {"1", "2", "3", "4"} and all(e["id"].startswith("P") for e in entries)
sc = S.make_script(p, cfg, entries, force=True)
segs = S.assemble(sc, p, cfg)
est = S.estimate_minutes(segs, cfg)
assert sc["angle"] == "list" and seen["gen"] == 1 and seen["expand"] >= 1
assert 5.85 <= est <= 8.35, est
assert sc["tags"][:3] == ["insulated water bottle", "best bottle for commuters", "commuter bottle"], sc["tags"]     # keywords first
assert len([t for t in sc["tags"] if t.lower() == "commuter bottle"]) == 1                                         # no duplicates
for needle in ("creator's opinion", "[product 3]", "CRITERIA", "number one"):
    assert needle in seen["check"], needle
assert any(w in segs[-1]["text"] for w in ("every", "Every", "each")), segs[-1]["text"]
md = S.script_to_md(sc, p)
assert "# ORDER: countdown" in md and "# PRODUCTS: 1=TestCo (TEST000001); 2=OtherCo" in md
assert [x["product"] for x in S.md_to_script(md)["segments"]] == [x["product"] for x in sc["segments"]]

# ---- ordinary products never get the list angle
single = load_project("TEST000001")
for forced in ("list", "comparison"):
    try:
        S.pick_angle(single, cfg, forced); raise AssertionError(forced)
    except SystemExit:
        pass

# ---- footage, overlays, chapters
script = S.md_to_script((p.dir / "script.md").read_text())
full = S.assemble(script, p, cfg)
arrays = V.synthesize_all(p, cfg, "mock", full)
tl = T.build_timeline(full, arrays, cfg, enforce=False)
shots = R.plan_shots(tl.rows, tl.total, entries, p.assets_dir, cfg, seed=p.asin, facts=p.facts)
for prod in ("1", "2", "3", "4"):
    owner = Counter()
    for r in tl.rows:
        if str(r.seg.get("product")) == prod:
            for sh in shots:
                if r.start <= sh.start < r.start + r.dur + r.pad:
                    owner[sh.asset["product"]] += 1
    share = owner[prod] / max(1, sum(owner.values()))
    print(f"  item #{prod}: {share:.0%} of its shots show product {prod}'s own footage")
    assert share >= 0.8, (prod, owner)
names = [x[1] for x in T.chapters(tl, c["labels"])]
for want in ("#4 FourthCo", "#3 ThirdCo", "#2 OtherCo", "#1 TestCo"):
    assert want in names, names
assert names.index("#4 FourthCo") < names.index("#1 TestCo"), "countdown: #4 is shown before #1"

W, H = 640, 360
ovs = build_overlays(tl.rows, tl.total, cfg, W, H, "Top 4", [], product_labels=c["labels"], kind="list",
                     rank_notes=[a or b for a, b in zip(c["rank_labels"], c["best_for"])])
first4 = next(r for r in tl.rows if str(r.seg.get("product")) == "4")
blank = np.full((H, W, 3), 120, np.uint8)
badge_zone = (slice(int(H * 0.30), int(H * 0.62)), slice(int(W * 0.04), int(W * 0.40)))


def badge_visible(t):
    fr = blank.copy()
    for o in ovs:
        o.apply(fr, t)
    return not np.array_equal(fr[badge_zone], blank[badge_zone])


assert badge_visible(first4.start + 1.2), "the big rank badge shows when an item starts"
assert not badge_visible(first4.start - 5.0), "...and not before it"
assert not badge_visible(first4.start + 6.0), "...and it leaves after a few seconds"

# ---- description, pinned comment, checklist
out = p.out_dir
F.write_description(p, cfg, sc, tl, {}, out / "description.txt")
desc = (out / "description.txt").read_text()
assert desc.splitlines()[0].startswith("As an Amazon Associate")
for i, (sub, lab) in enumerate(zip(c["products"], c["labels"]), 1):
    assert f"#{i} {lab} ({sub.facts['product_name']}): {sub.facts['affiliate_url']}" in desc, i
assert "Links to every product" in (out / "pinned_comment.txt").read_text()
F.write_review(p, sc, entries, ["n"], [], tl, Path("x.mp4"), "mock", out / "REVIEW_CHECKLIST.md")
rv = (out / "REVIEW_CHECKLIST.md").read_text()
assert "Products ranked (your order)" in rv and "ranking reason" in rv

# ---- length presets
cfg2 = load_config()
assert cli.choose_length(cfg2, single, None) == "review" and cfg2["length"]["target_minutes"] == 5.0
assert cli.choose_length(cfg2, single, "short") == "short" and cfg2["length"]["max_minutes"] == 4.8
for k in ("review", "compare", "list", "short"):
    assert load_config()["length_presets"][k]["max_minutes"] < 9, k
try:
    cli.choose_length(cfg2, single, "weird"); raise AssertionError("unknown length")
except SystemExit:
    pass
try:
    cli.main(["scenes", "LISTTEST01", "--no-vision"]); raise AssertionError("scenes are per product")
except SystemExit as e:
    assert "per product" in str(e)
print("OK - list videos: loading/limits, ranking prompts, keywords, countdown chapters, rank badges, footage matching, description, length presets")
