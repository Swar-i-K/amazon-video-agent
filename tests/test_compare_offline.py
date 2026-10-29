"""Offline test of comparison videos.   python tests/make_compare_fixture.py && python tests/test_compare_offline.py"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adagent import assets as A
from adagent import cli
from adagent import render as R
from adagent import scriptgen as S
from adagent import timeline as T
from adagent import tts as V
from adagent.common import load_config
from adagent.project import load_project, validate_facts

cfg = load_config()
p = load_project("CMPTEST001")
assert p.compare and p.compare["labels"] == ["TestCo", "OtherCo"], p.compare and p.compare["labels"]
assert p.compare["pick_ok"] is False                       # product 2 has only 2 hands-on notes
errs, warns = validate_facts(p)
assert not errs, errs
assert any("winner" in w or "hands_on" in w for w in warns), warns

entries = A.build_index_for(p, cfg, vision=False)
assert all(e["id"].startswith(("P1-", "P2-")) for e in entries) and {e["product"] for e in entries} == {"1", "2"}
assert all(Path(e["path"]).exists() for e in entries)

fy = S.facts_yaml(p)
assert "PRODUCT 1" in fy and "PRODUCT 2" in fy and "PICK_ALLOWED: false" in fy and "OtherCo" in fy
assert S.pick_angle(p, cfg, None) == "comparison"

# ---- generation with a fake model: compare prompts, product tags, expansion, fact-check rules
seen = {"gen": 0, "expand": 0, "check": None}


def fake_ask(model, system, user, **kw):
    if system == S.FACTCHECK_SYSTEM:
        seen["check"] = user
        return json.dumps({"issues": []})
    assert system == S.COMPARE_SYSTEM, "comparison projects must use the comparison writer"
    if "CURRENT SCRIPT" in user:
        seen["expand"] += 1
        assert '"product": "1|2|3|all"' in user
        words = int(re.search(r"about (\d+) words of NEW", user).group(1))
        new = [{"after_id": "5", "section": "product_1", "product": "1", "text": f"Extra detail {seen['expand']}-{i} about the first bottle in plain spoken words.",
                "onscreen": "", "asset_ids": ["P1-A01"], "source": "product", "priority": 3} for i in range(max(4, words // 24))]
        return json.dumps({"segments": new})
    seen["gen"] += 1
    assert "PRODUCT 1" in user and "PRODUCT 2" in user and "PICK_ALLOWED: false" in user and S.COMPARE_GUIDE in user
    assert "P1-A01" in user and "P2-A01" in user and "(OtherCo)" in user
    words = int(re.search(r"about (\d+) words", user).group(1))
    n = max(12, int(words * 0.30) // 22)
    secs = ["hook", "hook", "context", "overview", "product_1", "product_2", "head_to_head", "verdict"]
    segs = []
    for i in range(1, n + 1):
        sec = secs[min(len(secs) - 1, (i * len(secs)) // (n + 1))]
        prod = {"product_1": "1", "product_2": "2"}.get(sec, "all")
        segs.append({"id": i, "section": sec, "product": prod, "source": "product", "priority": 2, "onscreen": "",
                     "asset_ids": ["P1-A01"] if prod == "1" else ["P2-A01"] if prod == "2" else [],
                     "text": f"Line number {i} written in plain spoken words about this part of the comparison, nothing more."})
    return json.dumps({"title": "TestCo vs OtherCo", "description_intro": "x", "tags": ["a"], "segments": segs})


S.ask = fake_ask
cli.choose_length(cfg, p, None)                     # comparison videos use the 'compare' length preset (7.5 min aim)
assert (cfg["length"]["min_minutes"], cfg["length"]["max_minutes"]) == (5.5, 8.8)
sc = S.make_script(p, cfg, entries, force=True)
segs = S.assemble(sc, p, cfg)
est = S.estimate_minutes(segs, cfg)
assert sc["angle"] == "comparison" and seen["gen"] == 1 and seen["expand"] >= 1
assert cfg["length"]["min_minutes"] + 0.35 <= est <= cfg["length"]["max_minutes"] - 0.45, est
assert {s["product"] for s in sc["segments"]} >= {"1", "2", "all"}
assert "WRONG product" in seen["check"] and "[product 1]" in seen["check"] and "PICK_ALLOWED" in seen["check"]
assert segs[-1]["section"] == "outro" and any(w in segs[-1]["text"] for w in ("each", "both", "either", "Every link", "link"))

md = S.script_to_md(sc, p)
assert "# PRODUCTS: 1=TestCo (TEST000001); 2=OtherCo (TEST000002)" in md and "product: 1" in md
back = S.md_to_script(md)
assert [x["product"] for x in back["segments"]] == [x["product"] for x in sc["segments"]]

# comparison language is only a style warning, never a hard block
from adagent.project import check_text
pol, sty = check_text("The first one clearly beats the second.")
assert not pol and sty

# ---- ordinary products never get the comparison angle
single = load_project("TEST000001")
assert S.pick_angle(single, cfg, None) != "comparison"
try:
    S.pick_angle(single, cfg, "comparison")
    raise AssertionError("single product must refuse the comparison angle")
except SystemExit:
    pass

# ---- footage: lines about product N mostly show product N's footage
script = S.md_to_script((p.dir / "script.md").read_text())
full = S.assemble(script, p, cfg)
arrays = V.synthesize_all(p, cfg, "mock", full)
tl = T.build_timeline(full, arrays, cfg, enforce=False)
shots = R.plan_shots(tl.rows, tl.total, entries, p.assets_dir, cfg, seed=p.asin, facts=p.facts)
for prod in ("1", "2"):
    owner = Counter()
    for r in tl.rows:
        if str(r.seg.get("product")) == prod:
            for sh in shots:
                if r.start <= sh.start < r.start + r.dur + r.pad:
                    owner[sh.asset["product"]] += 1
    share = owner[prod] / max(1, sum(owner.values()))
    print(f"  lines about product {prod}: {share:.0%} of their shots show product {prod}'s footage")
    assert share >= 0.8, (prod, owner)

ch = T.chapters(tl, p.compare["labels"])
names = [c[1] for c in ch]
assert "TestCo" in names and "OtherCo" in names and "Head to head" in names, names

# ---- CLI behaviours
try:
    cli.main(["scenes", "CMPTEST001", "--no-vision"])
    raise AssertionError("scenes must refuse comparison projects")
except SystemExit as e:
    assert "per product" in str(e)

import yaml
cf = p.dir / "compare.yaml"
orig = cf.read_text()
cf.write_text(orig.replace("TEST000002", "NOPE000000"))
try:
    load_project("CMPTEST001")
    raise AssertionError("missing product folder must be reported")
except SystemExit as e:
    assert "NOPE000000" in str(e)
cf.write_text(orig.replace("products: [TEST000001, TEST000002]", "products: [TEST000001]"))
try:
    load_project("CMPTEST001")
    raise AssertionError("one product is not a comparison")
except SystemExit as e:
    assert "2 to 3" in str(e)
cf.write_text(orig)
other = load_project("TEST000002")
f2 = Path(other.dir / "facts.yaml")
f2o = f2.read_text()
f2.write_text(f2o.replace("https://amzn.to/test456", "https://amzn.to/test123"))
errs, _ = validate_facts(load_project("CMPTEST001"))
assert any("same affiliate_url" in e for e in errs), errs
f2.write_text(f2o)
print("OK - comparison: loading, validation, prompts, product tags, attribution rules, footage matching, chapters, guards")
