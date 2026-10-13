"""Offline test of the Groq layer with a simulated API:  python tests/test_llm_offline.py"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from adagent import llm

os.environ["GROQ_API_KEY"] = "test"
llm.time.sleep = lambda s: None
sent = []


class R:
    def __init__(self, code, js=None, text="", headers=None):
        self.status_code, self._js, self.text, self.headers = code, js, text or str(js), headers or {}

    def json(self):
        return self._js

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def ok(txt):
    return R(200, {"choices": [{"message": {"content": txt}}]})


seq = []


def fake_post(url, headers=None, json=None, timeout=0):
    import copy
    sent.append(copy.deepcopy(json))   # keep each attempt as it was sent
    return seq.pop(0)


def fake_get(url, headers=None, timeout=0):
    return R(200, {"data": [{"id": "openai/gpt-oss-120b"}, {"id": "qwen/qwen3-32b"}, {"id": "whisper-large-v3"},
                            {"id": "meta-llama/llama-4-maverick-17b-128e-instruct"}]})


llm.requests.post, llm.requests.get = fake_post, fake_get

# 1. rate limit then success
seq[:] = [R(429, text="slow down", headers={"retry-after": "1"}), ok('{"a":1}')]
assert llm.ask("llama-3.3-70b-versatile", "sys", "usr") == '{"a":1}'

# 2. configured model missing -> falls back to a model the account really has
llm._substitutes.clear()
seq[:] = [R(404, text='{"error":{"code":"model_not_found"}}'), ok("fine")]
assert llm.ask("llama-3.3-70b-versatile", "sys", "usr") == "fine"
assert sent[-1]["model"] == "openai/gpt-oss-120b", sent[-1]["model"]

# 3. vision request: image goes in user turn, vision fallback picks maverick
img = np.zeros((64, 64, 3), np.uint8)
seq[:] = [R(404, text="model_not_found"), ok('{"description":"x"}')]
assert llm.ask("meta-llama/llama-4-scout-17b-16e-instruct", "sys", "usr", images=[img]) == '{"description":"x"}'
body = sent[-1]
assert body["model"].endswith("maverick-17b-128e-instruct")
assert body["messages"][0]["role"] == "user" and body["messages"][0]["content"][1]["type"] == "image_url"

# 4. JSON-mode unsupported -> retried without response_format; max_tokens capped
llm._substitutes.clear()
seq[:] = [R(400, text="response_format json_object is not supported"), ok("{}")]
llm.ask("m", "s", "u", max_tokens=12000)
assert "response_format" not in sent[-1] and sent[-1]["max_tokens"] == 6500
# 5. the REAL account list from the user's log: vision must resolve to Qwen 3.8, text to gpt-oss-120b
real = ["allam-2-7b", "canopylabs/orpheus-arabic-saudi", "canopylabs/orpheus-v1-english", "meta-llama/llama-prompt-guard-2-22m",
        "meta-llama/llama-prompt-guard-2-86m", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b",
        "qwen/qwen3.8-27b", "whisper-large-v3", "whisper-large-v3-turbo"]
assert llm._pick_substitute("meta-llama/llama-4-scout-17b-16e-instruct", True, real) == "qwen/qwen3.8-27b"
assert llm._pick_substitute("llama-3.3-70b-versatile", False, real) == "openai/gpt-oss-120b"
assert llm._pick_substitute("x", True, ["openai/gpt-oss-120b"]) is None

# 6. a Qwen model that rejects reasoning options: steps down through simpler option sets until accepted
llm._substitutes.clear()
llm.requests.get = lambda url, headers=None, timeout=0: R(200, {"data": [{"id": i} for i in real]})
seq[:] = [R(400, text="unsupported value for reasoning_effort"), R(400, text="reasoning_format not supported"), ok('{"x":1}')]
assert llm.ask("qwen/qwen3.8-27b", "s", "u", images=[img]) == '{"x":1}'
last_body = sent[-1]
assert last_body["model"] == "qwen/qwen3.8-27b" and "reasoning_format" not in last_body and last_body.get("reasoning_effort") == "none"
assert sent[-3].get("reasoning_effort") == "none" and sent[-3].get("reasoning_format") == "hidden"
print("OK - Groq layer: retry on 429, model auto-fallback (text + vision), JSON-mode fallback, token cap, your real model list, reasoning-option step-down")
