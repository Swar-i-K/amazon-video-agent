"""Offline test: Groq daily-limit handling, saved photo checks, subscribe/like/share lines, line placement, removal.
    python tests/make_compare_fixture.py && python tests/test_limits_and_cta_offline.py"""
import copy
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import yaml

import adagent.llm as L
from adagent import assets as A
from adagent import scriptgen as S
from adagent.common import load_config
from adagent.project import check_text, load_project

cfg = load_config()
os.environ["GROQ_API_KEY"] = "test"
L.time.sleep = lambda s: None
posts, seq = [], []


class R:
    def __init__(self, code, text="", js=None, headers=None):
        self.status_code, self.text, self._js, self.headers = code, text, js, headers or {}

    def json(self):
        return self._js


def fake_post(url, headers=None, json=None, timeout=0):
    posts.append(copy.deepcopy(json))
    return seq.pop(0)


L.requests.post = fake_post
L._llm_cfg = lambda: {"provider": "groq"}      # no fallback models here: a daily limit on the only model must stop at once

# ---- 1. daily limit: stop at once, with the numbers; a per-minute limit still retries
TPD = ('{"error":{"message":"Rate limit reached for model `qwen/qwen3.8-27b` service tier `on_demand` on tokens per day (TPD): '
       'Limit 200000, Used 197621, Requested 2465. Please try again in 37.152s."}}')
seq[:] = [R(429, TPD)]
try:
    L.ask("qwen/qwen3.8-27b", "s", "u"); raise AssertionError("a daily limit must raise")
except L.DailyLimitError as e:
    assert "197,621 of 200,000" in str(e) and "24 hours" in str(e), str(e)
assert len(posts) == 1, "no retries after a daily limit"
L._exhausted.clear()
posts.clear()
seq[:] = [R(429, "Rate limit reached ... on tokens per minute (TPM): Limit 8000", headers={"retry-after": "1"}),
          R(200, js={"choices": [{"message": {"content": '{"a":1}'}, "finish_reason": "stop"}]})]
assert L.ask("openai/gpt-oss-120b", "s", "u") == '{"a":1}' and len(posts) == 2
print("  daily limit stops at once; per-minute limit still waits and retries: OK")

# ---- 2. photos are shrunk for the AI check
import numpy as np
sizes = []
L.jpeg_b64 = lambda im, max_side=1280: sizes.append(max_side) or "AAAA"
seq[:] = [R(200, js={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]})]
L.ask("qwen/qwen3.8-27b", "s", "u", images=[np.zeros((8, 8, 3), np.uint8)])
assert sizes == [640], sizes

# ---- 3. photo checks: saved after every photo, resumed after a stop, switch to turn the AI check off
p = load_project("TEST000001")
cache_file = p.cache_dir / "assets_index.json"
cache_file.unlink(missing_ok=True)
calls = []
OK = {"description": "a product photo", "shows_product": True, "has_human_face": False,
      "has_price_or_rating_ui": False, "has_third_party_logo_or_watermark": False}


def describe_then_limit(cfg, frames, kind):
    calls.append(1)
    if len(calls) > 3:
        raise A.DailyLimitError("Groq's daily limit for the model 'x' is used up.")
    return dict(OK)


A.describe = describe_then_limit
try:
    A.build_index(p, cfg, vision=True); raise AssertionError("must stop on the daily limit")
except SystemExit as e:
    msg = str(e)
    assert "3 of 10 files" in msg and "vision_check: false" in msg and "tomorrow" in msg, msg
assert len(json.loads(cache_file.read_text())) == 3, "progress saved after each photo"
calls.clear()
A.describe = lambda cfg, frames, kind: (calls.append(1), dict(OK))[1]
ents = A.build_index(p, cfg, vision=True)
assert len(calls) == 7 and len(ents) == 10, "resume: only the 7 unchecked files are described"
calls.clear()
A.build_index(p, cfg, vision=True)
assert calls == [], "nothing is described twice"
cfg_off = copy.deepcopy(cfg)
cfg_off["llm"]["vision_check"] = False
cache_file.unlink()
ents = A.build_index(p, cfg_off, vision=True)
assert calls == [] and all(e["usable"] for e in ents), "vision_check: false never calls the AI"
print("  photo checks saved per photo, resumed, and can be switched off: OK")

# ---- 4. subscribe intro, subscribe+like+share outro
p = load_project("TEST000001")
script = S.md_to_script((p.dir / "script.md").read_text())
segs = S.assemble(script, p, cfg)
assert segs[0]["section"] == "hook" and segs[1]["id"] == "S", [s["id"] for s in segs[:3]]
assert "subscribe" in segs[1]["text"].lower() and not re.search(r"\blike\b|\bshare\b", segs[1]["text"], re.I), "intro: subscribe only"
assert segs[-1]["section"] == "outro" and all(w in segs[-1]["text"].lower() for w in ("subscribe", "like", "share"))
c2 = copy.deepcopy(cfg); c2["cta"]["intro_position"] = "start"
assert S.assemble(script, p, c2)[0]["id"] == "S"
c3 = copy.deepcopy(cfg); c3["cta"]["intro"] = False
assert not any(s["id"] == "S" for s in S.assemble(script, p, c3))
c4 = copy.deepcopy(cfg); c4["disclosure"]["mode"] = "full"
order = [s["id"] for s in S.assemble(script, p, c4)]
assert order.index("S") < order.index("D"), "in full mode the disclosure still comes after the subscribe line"
own = copy.deepcopy(script)
own["segments"].insert(1, {"id": "i1", "section": "hook", "text": "Please subscribe if you haven't yet.", "onscreen": "",
                           "asset_ids": [], "source": "product", "priority": 1, "product": "all"})
assert not any(s["id"] == "S" for s in S.assemble(own, p, cfg)), "own subscribe line in the intro: do not add another"
own2 = copy.deepcopy(script)
own2["segments"].append({"id": "z1", "section": "verdict", "text": "Please subscribe and share this with a friend.", "onscreen": "",
                         "asset_ids": [], "source": "product", "priority": 1, "product": "all"})
last = S.assemble(own2, p, cfg)[-1]
assert last["section"] == "outro" and "subscribe" not in last["text"].lower(), last["text"]
for proj, name in (("CMPTEST001", "compare"),):
    q = load_project(proj)
    sq = S.md_to_script((q.dir / "script.md").read_text())
    t = S.assemble(sq, q, cfg)[-1]["text"].lower()
    assert all(w in t for w in ("subscribe", "like", "share")), (name, t)
print("  subscribe in the intro, subscribe+like+share in the outro, no double asks: OK")

# ---- 5. extra lines go to the right section, never under the verdict
q = load_project("CMPTEST001")
sq = S.md_to_script((q.dir / "script.md").read_text())
ents = A.build_index_for(q, cfg, vision=False)
last_id = sq["segments"][-1]["id"]
new = [{"after_id": last_id, "section": "verdict", "product": "1", "text": "Extra about the first bottle in plain spoken words."},
       {"after_id": last_id, "section": "", "product": "2", "text": "Extra about the second bottle in plain spoken words."},
       {"after_id": last_id, "section": "hook", "product": "all", "text": "Extra general comparison point in plain spoken words."}]
S.ask = lambda *a, **k: json.dumps({"segments": new})
before = len(sq["segments"])
out = S.expand(q, cfg, ents, sq, 100)
added = [s for s in out["segments"] if s["id"].startswith("x")]
assert len(added) == 3 and len(out["segments"]) == before + 3
sec = {re.search(r"first|second|general", s["text"]).group(0): s["section"] for s in added}
assert sec["first"] == "product_1" and sec["second"] == "product_2" and sec["general"] == "head_to_head", sec
assert [s["section"] for s in out["segments"] if s["section"] == "verdict"] == [s["section"] for s in sq["segments"] if s["section"] == "verdict"]
t1 = load_project("TEST000001")
s1 = S.md_to_script((t1.dir / "script.md").read_text())
S.ask = lambda *a, **k: json.dumps({"segments": [{"after_id": s1["segments"][-1]["id"], "section": "verdict", "text": "A general extra point for the viewer, said plainly."}]})
o1 = S.expand(t1, cfg, A.build_index(t1, cfg, vision=False), s1, 50)
x = [s for s in o1["segments"] if s["id"].startswith("x")][0]
assert x["section"] not in ("verdict", "hook"), x["section"]
print("  extra lines land in their own / their product's section, not the verdict: OK")

# ---- 6. lines that stay unsupported after a rewrite are removed and listed
calls = {"revise": 0}


def fake_ask(model, system, user, **kw):
    if system == S.FACTCHECK_SYSTEM:
        ids = re.findall(r"^\[(\d+)\]", user, re.M)
        return json.dumps({"issues": [{"id": "4", "problem": "unsupported removal method", "fix": "remove it"}] if "4" in ids else []})
    if "Rewrite ONLY these segments" in user:
        calls["revise"] += 1
        return json.dumps({"segments": [{"id": "4", "text": "Use a hair dryer to soften the adhesive before peeling it off.", "onscreen": ""}]})
    words = int(re.search(r"about (\d+) words", user).group(1))
    n = max(14, words // 22)
    segs = [{"id": i, "section": ["hook", "overview", "features", "in_use", "limits", "verdict"][min(5, i * 6 // (n + 1))], "source": "product",
             "priority": 2, "onscreen": "", "asset_ids": [], "text": f"Line number {i} in plain spoken words about this product, nothing more to say here."}
            for i in range(1, n + 1)]
    segs[3]["text"] = "Use a hair dryer to soften the adhesive before peeling it off."
    return json.dumps({"title": "T", "description_intro": "x", "tags": [], "segments": segs})


S.ask = fake_ask
cfg5 = load_config()
cfg5["length"].update(min_minutes=1.0, max_minutes=8.0, target_minutes=5.0)
sc = S.make_script(t1, cfg5, A.build_index(t1, cfg5, vision=False), angle="honest_pros_cons", force=True)
assert all(s["id"] != "4" for s in sc["segments"]), "unsupported line removed"
assert sc["review"]["removed"][0]["id"] == "4" and "hair dryer" in sc["review"]["removed"][0]["text"] and calls["revise"] >= 1
md = S.script_to_md(sc, t1)
assert "REMOVED (unsupported, taken out of the script) [4]" in md
print("  unsupported lines are removed (and listed) instead of left in the script: OK")

# ---- 7. never read the notes aloud
pol, _ = check_text("Hands-on notes mention the heavy-duty film is sturdy.")
assert pol and "notes" in pol[0]
assert check_text("In my use the film stayed tight, and the brand says it is heavy duty.")[0] == []
for sysmsg in (S.SYSTEM, S.COMPARE_SYSTEM, S.LIST_SYSTEM):
    assert "NEVER say 'hands-on notes'" in sysmsg
print("  'hands-on notes' is never spoken: OK")

# ---- 8. workflow keeps caches for ALL projects, even after a failed run
wf = yaml.safe_load(open(Path(__file__).resolve().parent.parent / ".github/workflows/make-video.yml"))
steps = wf["jobs"]["build"]["steps"]
rest = next(x for x in steps if str(x.get("uses", "")).startswith("actions/cache/restore"))
save = next(x for x in steps if str(x.get("uses", "")).startswith("actions/cache/save"))
assert rest["with"]["path"] == "projects/*/.cache" and save["with"]["path"] == "projects/*/.cache" and save["if"] == "always()"
assert [x.get("name") for x in steps].index(save["name"]) > [x.get("name") or x.get("uses") for x in steps].index("Run")
print("OK - limits & CTA: daily-limit stop, saved/resumed photo checks, vision switch, subscribe/like/share, line placement, removal, no notes aloud, workflow caches")

# ---- 9. a key written twice is reported, never silently overridden
from adagent.common import load_config as _lc
cfgdup = Path(__file__).resolve().parent.parent / "config.yaml"
orig = cfgdup.read_text()
try:
    cfgdup.write_text(orig + "\ncta:\n  intro: false\n")
    try:
        _lc(); raise AssertionError("duplicate top-level key must be refused")
    except SystemExit as e:
        assert "appears twice" in str(e) and "line" in str(e), str(e)
finally:
    cfgdup.write_text(orig)
d = Path(__file__).resolve().parent.parent / "projects" / "TESTDUP001"
d.mkdir(parents=True, exist_ok=True)
(d / "facts.yaml").write_text("asin: TESTDUP001\nproduct_name: X\nhands_on_notes:\n  - a\nspecs:\n  - s\nhands_on_notes:\n  - b\n")
try:
    try:
        load_project("TESTDUP001"); raise AssertionError("duplicate hands_on_notes must be refused")
    except SystemExit as e:
        assert "hands_on_notes" in str(e) and "Merge the two" in str(e), str(e)
finally:
    (d / "facts.yaml").unlink(); d.rmdir()
print("  a key written twice is reported with its line number: OK")
