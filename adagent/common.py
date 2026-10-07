def extract_json(text: str):
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    t = re.sub(r"```(?:json)?", "", t).strip()
    starts = [i for i in (t.find("{"), t.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"No JSON found in model reply: {t[:300]!r}")
    t = t[min(starts): max(t.rfind("}"), t.rfind("]")) + 1]

    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    try:  # repairs trailing commas, single quotes, unquoted keys, etc.
        from json_repair import repair_json
        return json.loads(repair_json(t))
    except Exception:
        pass
    t2 = re.sub(r",\s*([}\]])", r"\1", t)  # fallback: just strip trailing commas
    try:
        return json.loads(t2)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON ({e}). Raw reply start: {t[:500]!r}")
