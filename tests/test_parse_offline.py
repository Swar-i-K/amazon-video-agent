"""Offline test: the script writer survives bad model replies.   python tests/test_parse_offline.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adagent import assets as A
from adagent import scriptgen as S
from adagent.common import extract_json, load_config
from adagent.project import load_project

# extract_json never returns a bare string
for bad in ['"just a string"', "no json at all", '{"a": 1', ""]:
    try:
        r = extract_json(bad)
        assert isinstance(r, (dict, list)), (bad, r)
    except ValueError:
        pass

segs = [{"id": i, "section": "features", "text": f"Line number {i} about the towel in plain words.", "onscreen": "",
         "asset_ids": [], "source": "product", "priority": 2} for i in range(1, 41)]
full = json.dumps({"title": "T", "description_intro": "x", "tags": [], "segments": segs})
truncated = full[: full.rfind('{"id": 38')]             # reply cut off mid-way
assert len(S.salvage_segments(truncated)) == 37
assert S.parse_script_reply(truncated)["salvaged"]
assert S.parse_script_reply('"a plain string"') is None
assert S.parse_script_reply("```json\n" + full + "\n```")["segments"]
assert S.parse_script_reply(json.dumps(segs))["segments"]   # bare list

cfg = load_config()
p = load_project("TEST000001")
entries = A.build_index(p, cfg, vision=False)
replies = ['"here is your script"', truncated, full]
asked = []


def fake_ask(model, system, user, **kw):
    asked.append(user)
    return replies.pop(0)


S.ask = fake_ask
out = S.generate(p, cfg, entries, "first_impressions", 900)
assert len(out["segments"]) == 37 and len(asked) == 2, (len(out["segments"]), len(asked))
assert "cut off or was not valid JSON" in asked[1]

replies[:] = ['"nope"', "still nothing", ""]
try:
    S.generate(p, cfg, entries, "first_impressions", 900)
    raise AssertionError("should have failed after 3 bad replies")
except RuntimeError as e:
    assert "usable script" in str(e)

S.ask = lambda *a, **k: "total garbage"                  # unreadable fact-check must not crash
iss = S.factcheck(p, cfg, {"segments": out["segments"]})
assert iss and iss[0]["id"] == "*"
print("OK - bad replies are retried, truncated replies are salvaged, fact-check failures are flagged not fatal")
