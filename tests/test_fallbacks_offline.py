"""Offline test: fallback models on a daily limit, 'request too large', matched-words policy message, workflow log/cleaner.
    python tests/make_fixture.py && python tests/test_fallbacks_offline.py"""
import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import yaml

from adagent import cli, llm

os.environ["GROQ_API_KEY"] = "test"
slept = []
llm.time.sleep = lambda s: slept.append(s)
llm._llm_cfg = lambda: {"provider": "groq", "vision_fallbacks": ["fb/vision-b", "fb/vision-c"], "text_fallbacks": ["fb/text-b"]}

TPD = ("Rate limit reached for model `m` in organization on tokens per day (TPD): "
       "Limit 200000, Used 197621, Requested 2465. Please try again in 38s.")


class R:
    def __init__(self, code, js=None, text="", headers=None):
        self.status_code, self._js, self.text, self.headers = code, js, text or str(js), headers or {}

    def json(self):
        return self._js


def ok(txt):
    return R(200, {"choices": [{"message": {"content": txt}, "finish_reason": "stop"}]})


sent, seq = [], []


def fake_post(url, headers=None, json=None, timeout=0):
    sent.append(copy.deepcopy(json))
    return seq.pop(0)


llm.requests.post = fake_post
img = np.zeros((64, 64, 3), np.uint8)
models = lambda: [b["model"] for b in sent]

# 1. daily limit on the main model: straight to the fallback, no waiting
seq[:] = [R(429, text=TPD), ok('{"a":1}')]
assert llm.ask("main/vision-a", "s", "u", images=[img]) == '{"a":1}'
assert models() == ["main/vision-a", "fb/vision-b"] and not slept and "main/vision-a" in llm._exhausted
# 2. an exhausted model is not tried again
sent.clear(); seq[:] = [ok('{"a":2}')]
llm.ask("main/vision-a", "s", "u", images=[img])
assert models() == ["fb/vision-b"]
# 3. a fallback name your account does not have: the agent looks at your account's model list; a substitute that is also used up
#    is a daily limit again, so it moves on to the next fallback (it never stops the run)
llm.requests.get = lambda *a, **k: R(200, {"data": [{"id": "main/vision-a"}, {"id": "fb/vision-c"}]})
llm._exhausted.clear(); llm._substitutes.clear(); sent.clear()
seq[:] = [R(429, text=TPD),                                                    # main/vision-a: daily limit
          R(404, text='{"error":{"code":"model_not_found","message":"The model `fb/vision-b` does not exist"}}'),
          R(429, text=TPD),                                                    # the substitute picked from your list is used up too
          ok('{"a":3}')]                                                       # fb/vision-c works
assert llm.ask("main/vision-a", "s", "u", images=[img]) == '{"a":3}' and models()[-1] == "fb/vision-c", models()
# 4. every model used up: one clear DailyLimitError naming all of them
llm._exhausted.clear(); sent.clear()
seq[:] = [R(429, text=TPD), R(429, text=TPD), R(429, text=TPD)]
try:
    llm.ask("main/vision-a", "s", "u", images=[img]); raise AssertionError("must raise")
except llm.DailyLimitError as e:
    assert "main/vision-a" in str(e) and "fb/vision-c" in str(e) and "24 hours" in str(e), str(e)
# 5. text calls use text_fallbacks; an ordinary error on the FIRST model is reported, not hidden by a fallback
llm._exhausted.clear(); sent.clear(); seq[:] = [R(429, text=TPD), ok('{"t":1}')]
assert llm.ask("main/text", "s", "u") == '{"t":1}' and models() == ["main/text", "fb/text-b"]
llm._exhausted.clear(); sent.clear(); seq[:] = [R(401, text="invalid api key")]
try:
    llm.ask("main/text2", "s", "u"); raise AssertionError("must raise")
except RuntimeError as e:
    assert not isinstance(e, llm.DailyLimitError) and models() == ["main/text2"]
print("  fallback models: failover, skipped when used up, bad names skipped, all-exhausted message, ordinary errors still reported: OK")

# 6. 'request too large': lower max_tokens at once (no waiting); an input-side one fails fast
OTPM = ("Request too large for model `q` in organization `o` service tier `on_demand` on output tokens per minute (OTPM): "
        "Limit 1000, Requested 1316. The request's expected output tokens exceed the enforced limit; reduce max_tokens")
slept.clear(); sent.clear(); seq[:] = [R(429, text=OTPM), ok('{"x":1}')]
assert llm._ask_groq("size/model", "s", "u", [img], 1500, 0.0) == '{"x":1}' and not slept
assert len(sent) == 2 and sent[1]["max_tokens"] < 1000 and sent[1]["max_tokens"] < sent[0]["max_tokens"]
sent.clear(); seq[:] = [R(429, text="Request too large for model x on tokens per minute (TPM): Limit 6000, Requested 9000.")]
try:
    llm._ask_groq("size/model2", "s", "u", [img], 300, 0.0); raise AssertionError("must raise")
except RuntimeError as e:
    assert "waiting will not help" in str(e) and len(sent) == 1
print("  'request too large': max_tokens lowered at once / input-side fails fast: OK")

# 7. a blocked render shows the exact matched words and the line
sp = ROOT / "projects" / "TEST000001" / "script.md"
orig = sp.read_text()
try:
    sp.write_text(orig.replace("## hook", "## hook\n[p1] It costs $19.99 today, a real bargain. | p: 1", 1))
    try:
        cli.main(["render", "TEST000001", "--voice", "mock", "--no-vision"]); raise AssertionError("price line must block the render")
    except SystemExit as e:
        m = str(e)
        assert "[p1]" in m and "matched:" in m and "$19.99" in m and "text: It costs $19.99 today" in m, m
finally:
    sp.write_text(orig)
print("  policy block shows the matched words and the text: OK")

# 8. workflow: log + error box + 3-day downloads + script saved after a failed render; the cleaner
wf = yaml.safe_load((ROOT / ".github/workflows/make-video.yml").read_text())["jobs"]["build"]["steps"]
by = {s.get("name"): s for s in wf}
run = by["Run"]
assert run["id"] == "run" and "pipefail" in run["run"] and "tee run.log" in run["run"]
summ = by["Show the result on the run summary page"]
assert summ["if"] == "always()" and "GITHUB_STEP_SUMMARY" in summ["run"] and "::error" in summ["run"] and "steps.run.outcome" in summ["run"]
up = by["Upload results"]
assert "run.log" in up["with"]["path"] and up["with"]["retention-days"] == 3
assert by["Save script + anti-duplication history"]["if"] == "always()"
cl = yaml.safe_load((ROOT / ".github/workflows/clean-artifacts.yml").read_text())
assert cl["permissions"] == {"actions": "write"} and cl[True]["workflow_dispatch"]["inputs"]["older_than_days"]["default"] == "1"
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["llm"]
assert cfg["vision_fallbacks"] and cfg["text_fallbacks"]
print("OK - merged fixes: fallback models, request-too-large, matched-words message, run log + error box, 3-day downloads, script saved on failure, artifact cleaner")
