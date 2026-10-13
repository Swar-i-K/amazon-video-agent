"""Offline tests for the tolerant JSON parser, Groq retry/fallback logic and friendly YAML errors.
    python tests/test_llm_and_json.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from adagent.common import extract_json
import adagent.llm as L

# ---------- messy model output ----------
cases = {
    "plain": '{"a": 1}',
    "fenced": 'Sure! Here you go:\n```json\n{"a": 1, "b": [1,2]}\n```\nHope that helps',
    "think": '<think>let me reason {"x": 0}</think>\n{"a": 1}',
    "trailing commas": '{"a": [1, 2,], "b": {"c": 3,},}',
    "comments": '{"a": 1, // note\n "b": 2 /* more */}',
    "newline in string": '{"text": "line one\nline two"}',
    "curly delimiters": '{\u201ca\u201d: \u201cb\u201d}',
    "chatter + array": 'Result: [ {"id": 1}, {"id": 2} ] done',
    "comma inside string kept": '{"t": "Hello, world, and more,",}',
}
for name, raw in cases.items():
    out = extract_json(raw)
    assert out is not None, name
assert extract_json(cases["trailing commas"]) == {"a": [1, 2], "b": {"c": 3}}
assert extract_json(cases["comma inside string kept"])["t"] == "Hello, world, and more,"
assert extract_json(cases["think"]) == {"a": 1}
assert extract_json(cases["newline in string"])["text"] == "line one\nline two"
try:
    extract_json('{"a": "cut off')
    raise SystemExit("should have failed")
except ValueError as e:
    assert "cut off" in str(e) or "not valid" in str(e)
print("extract_json: OK (%d messy cases)" % len(cases))

# ---------- Groq client with a fake network ----------
class R:
    def __init__(self, code, body, headers=None):
        self.status_code, self._b, self.headers, self.text = code, body, headers or {}, json.dumps(body)
    def json(self): return self._b

posts = []
def fake_post(url, headers=None, json=None, timeout=0):
    posts.append(json)
    n = len(posts)
    if n == 1:
        return R(429, {"error": "rate"}, {"retry-after": "0"})
    if json["model"] == "llama-3.3-70b-versatile":
        return R(404, {"error": {"code": "model_not_found", "message": "model does not exist"}})
    if "response_format" in json and json["model"] == "openai/gpt-oss-120b" and not getattr(fake_post, "jm", False):
        fake_post.jm = True
        return R(400, {"error": {"message": "json mode not supported for this model"}})
    return R(200, {"choices": [{"message": {"content": '{"ok": true}'}}]})

def fake_get(url, headers=None, timeout=0):
    return R(200, {"data": [{"id": "whisper-large-v3"}, {"id": "openai/gpt-oss-120b"}, {"id": "llama-3.1-8b-instant"},
                            {"id": "meta-llama/llama-4-scout-17b-16e-instruct"}]})

L.requests.post, L.requests.get = fake_post, fake_get
L.time.sleep = lambda s: None
import os
os.environ["GROQ_API_KEY"] = "test"
L._llm_cfg.cache_clear()
txt = L.ask("llama-3.3-70b-versatile", "sys", "user", max_tokens=12000)
assert json.loads(txt) == {"ok": True}
models_used = [p["model"] for p in posts]
assert models_used[-1] == "openai/gpt-oss-120b", models_used          # fell back automatically
assert all(p["max_tokens"] <= 6500 for p in posts)                      # capped for free-tier limits
assert "response_format" not in posts[-1]                               # dropped after the JSON-mode 400
print("Groq: 429 retry, model_not_found fallback, JSON-mode fallback, token cap: OK ->", models_used)

# vision request shape + vision fallback
posts.clear()
def vpost(url, headers=None, json=None, timeout=0):
    posts.append(json)
    if json["model"] == "bad-vision":
        return R(404, {"error": {"message": "The model `bad-vision` does not exist"}})
    return R(200, {"choices": [{"message": {"content": '{"description": "x"}'}}]})
L.requests.post = vpost
L._groq_models = None
img = np.zeros((64, 64, 3), np.uint8)
L.ask("bad-vision", "sys", "describe", images=[img])
last = posts[-1]
assert last["model"] == "meta-llama/llama-4-scout-17b-16e-instruct"
parts = last["messages"][0]["content"]
assert parts[0]["type"] == "text" and parts[1]["type"] == "image_url" and parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
assert len(last["messages"]) == 1  # no separate system message with images
print("Groq vision: request shape + vision model fallback: OK")

# ---------- friendly YAML error ----------
from adagent.project import load_project
d = Path(__file__).resolve().parent.parent / "projects" / "TESTYAML01"
d.mkdir(parents=True, exist_ok=True)
(d / "facts.yaml").write_text('asin: TESTYAML01\nproduct_name: "Cooling Towels 47"x12" pack"\nspecs: [a, b]\n  - c\n')
try:
    load_project("TESTYAML01")
    raise SystemExit("should have failed")
except SystemExit as e:
    assert "formatting error" in str(e) and "line" in str(e), e
    print("YAML error message: OK ->", str(e).splitlines()[0])
finally:
    (d / "facts.yaml").unlink(); d.rmdir()
print("ALL OK")

# ---------- a ```yaml code-box pasted from chat is ignored, not fatal ----------
d = Path(__file__).resolve().parent.parent / "projects" / "TESTYAML02"
d.mkdir(parents=True, exist_ok=True)
(d / "facts.yaml").write_text("```yaml\nasin: TESTYAML02\nproduct_name: Fenced Bottle\nspecs:\n  - a\n  - b\n```\n")
try:
    pj = load_project("TESTYAML02")
    assert pj.facts["product_name"] == "Fenced Bottle" and pj.facts["specs"] == ["a", "b"], pj.facts
    print("pasted ```yaml lines are ignored: OK")
finally:
    (d / "facts.yaml").unlink(); d.rmdir()
