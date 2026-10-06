"""Offline test of the script pipeline with a fake LLM (no API key needed):
    python tests/test_scriptgen_offline.py     (run tests/make_fixture.py first)
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adagent import assets as A
from adagent import scriptgen as S
from adagent.common import load_config
from adagent.project import check_text, load_project

calls = {"gen": 0, "check": 0, "revise": 0}


def fake_ask(model, system, user, images=None, max_tokens=0, temperature=0):
    if system == S.SYSTEM and "Return JSON in this shape" in user:
        calls["gen"] += 1
        words = int(re.search(r"about (\d+) words", user).group(1))
        if calls["gen"] == 1:
            words = int(words * 0.45)          # first draft far too short -> must trigger a retry
        n = max(6, words // 22)
        secs = ["hook", "hook"] + ["context"] * 3 + ["features"] * (n - 9) + ["limits"] * 2 + ["verdict"] * 2
        segs = []
        for i, sec in enumerate(secs[:n], 1):
            text = ("The brand says the lid closes with a quarter turn and that matters on a busy morning commute " * 2)[:130]
            text += f" Point number {i} in plain words, nothing more to add here honestly."
            if i == 7:
                text = "Right now it costs $24 which is a great deal."     # must be caught by policy rules
            segs.append({"id": i, "section": sec, "text": text, "onscreen": "", "asset_ids": ["A01"], "source": "product", "priority": 1 + i % 3})
        return json.dumps({"title": "Test", "description_intro": "x", "tags": ["a"], "segments": segs})
    if system == S.FACTCHECK_SYSTEM:
        calls["check"] += 1
        if calls["check"] == 1:
            return json.dumps({"issues": [{"id": "3", "problem": "unsupported claim", "fix": "remove it"}]})
        return json.dumps({"issues": []})
    if "Rewrite ONLY these segments" in user:
        calls["revise"] += 1
        return json.dumps({"segments": [{"id": "3", "text": "Rewritten and now supported by the facts supplied.", "onscreen": ""}]})
    raise AssertionError("unexpected call")


S.ask = fake_ask
cfg = load_config()
p = load_project("TEST000001")
entries = A.build_index(p, cfg, vision=False)
sc = S.make_script(p, cfg, entries, angle="honest_pros_cons", force=True)
rv = sc["review"]
segs = S.assemble(sc, p, cfg)
est = S.estimate_minutes(segs, cfg)
print("generations:", calls, "| estimated minutes:", round(est, 2), "| segments:", len(segs))
assert calls["gen"] >= 2, "short first draft should have forced a regeneration"
assert calls["revise"] == 1 and any(s["text"].startswith("Rewritten") for s in sc["segments"])
assert 5.3 <= est <= 8.6, est
assert any(i["id"] == "7" for i in rv["policy"]), "price line must be flagged"
assert segs[0]["section"] == "hook" and any(s["section"] == "disclosure" for s in segs) and segs[-1]["section"] == "outro"

md = S.script_to_md(sc, p)
back = S.md_to_script(md)
assert len(back["segments"]) == len(sc["segments"]) and back["angle"] == "honest_pros_cons"
assert back["segments"][1]["text"] == sc["segments"][1]["text"]
pol, _ = check_text("It is only $19.99 today")
assert pol
trimmed = S.trim_to_length(sc["segments"] * 2, cfg, 7.0)
assert S.estimate_minutes(trimmed, cfg) <= 7.0 or len(trimmed) < len(sc["segments"] * 2)
print("OK - script pipeline logic works (policy flag, fact-check/revise, length retry, md round-trip)")
