"""Shared helpers."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


class _StrictLoader(yaml.SafeLoader):
    """YAML loader that refuses a key written twice (the second one would silently replace the first)."""


def _no_duplicate_keys(loader, node, deep=False):
    seen = set()
    for knode, _ in node.value:
        key = loader.construct_object(knode, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(None, None, f"the line '{key}:' appears twice (the second would replace the first)",
                                                    knode.start_mark)
        seen.add(key)
    return yaml.constructor.SafeConstructor.construct_mapping(loader, node, deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys)


def load_yaml_strict(text: str):
    return yaml.load(text, Loader=_StrictLoader)


def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    try:
        return load_yaml_strict(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f" (line {mark.line + 1})" if mark is not None else ""
        raise SystemExit(f"config.yaml has a formatting error{where}: {getattr(e, 'problem', None) or e}. "
                         "Fix that line on GitHub (pencil icon); the spaces at the start of a line matter.")


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list, **kw) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{r.stderr[-1500:]}")
    return r


def ffprobe(path) -> dict:
    r = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    return json.loads(r.stdout)


def media_duration(path) -> float:
    return float(ffprobe(path)["format"]["duration"])


def video_info(path) -> dict:
    info = ffprobe(path)
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = (vs.get("avg_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    dur = float(info["format"].get("duration") or vs.get("duration") or 0)
    rot = 0
    for sd in vs.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = rot or int((vs.get("tags") or {}).get("rotate", 0) or 0)
    w, h = int(vs["width"]), int(vs["height"])
    if abs(rot) % 180 == 90:  # OpenCV applies rotation metadata automatically
        w, h = h, w
    return {"width": w, "height": h, "fps": fps or 30.0, "duration": dur}


def sha1_file(path, limit: int = 4_000_000) -> str:
    h = hashlib.sha1()
    p = Path(path)
    h.update(str(p.stat().st_size).encode())
    with open(p, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()[:16]


def sha1_text(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:20]


def _clean_json(s: str) -> str:
    """Strip // and /* */ comments and trailing commas that sit OUTSIDE of strings."""
    out, i, n = [], 0, len(s)
    in_str = esc = False
    while i < n:
        c = s[i]
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
        elif c == "/" and s[i:i + 2] == "//":
            while i < n and s[i] != "\n":
                i += 1
            continue
        elif c == "/" and s[i:i + 2] == "/*":
            j = s.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        elif c == ",":
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            if j < n and s[j] in "}]":
                i += 1
                continue
            out.append(c)
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _loads_lenient(txt: str):
    """json.loads that tolerates comments, trailing commas, raw line breaks inside strings, curly quotes and single-quoted pseudo-JSON."""
    straight = txt.replace("\u201c", '"').replace("\u201d", '"')
    variants = [txt, _clean_json(txt), straight, _clean_json(straight)]
    for v in variants:
        try:
            return json.loads(v, strict=False)
        except Exception:
            pass
    try:  # optional helper, used only if installed
        import json_repair
        return json_repair.loads(txt)
    except Exception:
        pass
    import ast
    return ast.literal_eval(variants[1])                           # single quotes, True/False


def _is_container(x) -> bool:
    return isinstance(x, (dict, list))


def extract_json(text: str):
    """Pull the first JSON object/array out of an LLM reply (tolerates code fences, <think> blocks, sloppy JSON).

    Always returns a dict or list; raises ValueError otherwise (so callers can retry instead of crashing later)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("No JSON found in model reply")
    body = text[min(starts):]
    try:
        obj, _ = json.JSONDecoder(strict=False).raw_decode(body)
        if _is_container(obj):
            return obj
    except Exception:
        pass
    end = max(body.rfind("}"), body.rfind("]"))
    try:
        obj = _loads_lenient(body[: end + 1])
    except Exception as e:
        raise ValueError(f"Model reply was not valid JSON ({type(e).__name__})") from e
    if not _is_container(obj):
        raise ValueError("Model reply was not a JSON object")
    return obj


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9'’\-]+", text))


def fmt_ts(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"
