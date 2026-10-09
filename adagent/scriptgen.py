"""Write a natural, fact-bound narration script, check it, and keep it within 5-9 minutes.

Changes in this version:
- No affiliate disclosure is spoken (you add it to the description and pinned comment yourself).
- Your own intro line (after the hook) and outro lines (links note, like/subscribe/comment, thanks) are
  added to the script itself, so they appear in script.md where you can edit them.
"""
from __future__ import annotations

import datetime as dt
import json
import random
import re

import yaml

from .common import ROOT, extract_json, log, word_count
from .llm import ask
from .project import Project, check_text

ANGLES = {
    "first_impressions": "A hands-on first look: what it is, what stands out, and what a buyer should know before deciding.",
    "who_its_for": "A decision guide: who this suits, who should skip it, and how to decide for yourself.",
    "problem_solution": "Open with the everyday problem this product addresses, then walk through how each feature deals with it.",
    "buyers_checklist": "Teach the viewer what to check when choosing this kind of product, then measure this one against that checklist.",
    "day_in_the_life": "Walk through realistic situations where it gets used, tying each moment to a supplied fact.",
    "honest_pros_cons": "A balanced rundown that takes the downsides as seriously as the upsides.",
    "gift_angle": "Frame it around who it would suit as a gift and what to consider before gifting it.",
}

SECTION_LABELS = {
    "hook": "Intro", "disclosure": "Intro", "context": "What to look for", "overview": "Overview",
    "features": "Key features", "in_use": "In use", "limits": "Limitations",
    "who_for": "Who it suits", "who_skip": "Who should skip it", "verdict": "Verdict", "outro": "Verdict",
}

SYSTEM = """You write narration for faceless product videos. A human creator reviews every script, an AI voice reads it, and the creator posts it.
Your narration must sound like one real person talking to one viewer: contractions, varied sentence length (some very short), concrete detail, natural transitions, the occasional honest aside. No sales-pitch voice.

HARD RULES
1. Product facts come ONLY from FACTS. Never invent or infer specs, materials, numbers, features, certifications, compatibility, durability, or results. If a detail is not in FACTS, do not state it.
2. Items under manufacturer_claims are the brand's words: attribute them ("the brand says", "according to the listing"). Do not present them as your own findings.
3. First-person experience ("I", "in my use") is allowed ONLY when supported by hands_on_notes. Otherwise stay neutral ("if you're the kind of person who...").
4. General guidance about the product category (what to look for, common trade-offs) is allowed, but it must not assert anything about this specific product. Mark such segments source = "general".
5. NEVER mention: prices, discounts, deals, coupons, limited-time offers, star ratings, review counts, what customers/reviewers say, "best seller", "Amazon's Choice", other retailers, health or medical claims, guarantees.
6. Avoid: game-changer, revolutionary, must-have, life-changing, perfect for everyone, "in today's video", "smash that like". No exclamation-mark hype.
7. Do not write a disclosure, a subscribe or like request, a "links in the description" line, or a sign-off. The intro and outro lines are added automatically.
8. Write for the ear. Each segment is 1-3 sentences, 14-32 words. Spell out units ("20 ounces"). No emojis, no markdown, no abbreviations a voice would mispronounce.
9. onscreen is optional: at most 4 words, and only a fact that appears in FACTS (e.g. "20 oz capacity"). Otherwise "".
10. asset_ids: pick 1-3 ids from AVAILABLE FOOTAGE that match what the narration is talking about. Use [] if nothing fits. Never describe something the footage does not show.
11. Output ONLY JSON, no commentary."""

SECTION_GUIDE = (
    "Section order: hook (2-3 segments, no greeting, start in the middle of a thought), context, overview, "
    "features (several segments, one idea each), in_use, limits (be honest, use cons/not_for if given), "
    "who_for, who_skip, verdict. Skip a section only if FACTS give nothing real to say about it."
)

SCHEMA = """{
 "title": "YouTube title, under 70 chars, no clickbait, no price",
 "description_intro": "2-3 plain sentences describing the video, no hype, no price",
 "tags": ["8-12 relevant tags"],
 "segments": [
  {"id": 1, "section": "hook|context|overview|features|in_use|limits|who_for|who_skip|verdict",
   "text": "narration", "onscreen": "", "asset_ids": ["A01"], "source": "product|handson|general", "priority": 1}
 ]
}
priority: 1 = essential, 2 = useful, 3 = nice to have (these are cut first if the video runs long)."""


# ---------------------------------------------------------------------------
def facts_yaml(project: Project) -> str:
    keep = ["product_name", "brand", "category", "audience", "manufacturer_claims", "specs",
            "hands_on_notes", "pros", "cons", "not_for", "use_cases", "notes"]
    d = {k: project.facts[k] for k in keep if project.facts.get(k)}
    return yaml.safe_dump(d, sort_keys=False, allow_unicode=True, width=1000)


def assets_block(entries: list[dict]) -> str:
    lines = []
    for e in entries:
        if e["usable"]:
            extra = f", {e['duration']:.0f}s" if e["kind"] == "clip" else ""
            lines.append(f"{e['id']} [{e['kind']}{extra}] {e['description']}")
    return "\n".join(lines) or "(none)"


def estimate_minutes(segments: list[dict], cfg: dict) -> float:
    ln, v = cfg["length"], cfg["voice"]
    words = sum(word_count(s["text"]) for s in segments)
    secs = words / (ln["wpm"] / 60.0) + len(segments) * v["pause_sentence"]
    prev = None
    for s in segments:
        if prev and s.get("section") != prev:
            secs += v["pause_section"]
        prev = s.get("section")
    return (secs + ln["lead_in"] + ln["tail"]) / 60.0


def pick_angle(project: Project, cfg: dict, forced: str | None) -> str:
    if forced:
        if forced not in ANGLES:
            raise SystemExit(f"Unknown angle '{forced}'. Choose from: {', '.join(ANGLES)}")
        return forced
    if project.facts.get("angle") in ANGLES:
        return project.facts["angle"]
    recent = [e.get("angle") for e in load_archive()[-8:]]
    pool = sorted(ANGLES, key=lambda a: (recent.count(a), a))
    least = [a for a in pool if recent.count(a) == recent.count(pool[0])]
    return random.Random(project.asin).choice(least)


# ----------------------------- archive / anti-duplication ------------------
ARCHIVE = ROOT / "scripts_archive"


def load_archive() -> list[dict]:
    out = []
    for p in sorted(ARCHIVE.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            pass
    return sorted(out, key=lambda e: e.get("date", ""))


def save_archive(project: Project, script: dict) -> None:
    ARCHIVE.mkdir(exist_ok=True)
    text = " ".join(s["text"] for s in script["segments"])
    (ARCHIVE / f"{project.asin}.json").write_text(json.dumps({
        "asin": project.asin, "date": dt.date.today().isoformat(),
        "angle": script.get("angle"), "title": script.get("title"), "text": text}, indent=1))


def shingles(text: str, n: int = 4) -> set:
    w = re.findall(r"[a-z0-9']+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def max_similarity(text: str, asin: str) -> tuple[float, str]:
    a = shingles(text)
    best, who = 0.0, ""
    for e in load_archive():
        if e.get("asin") == asin:
            continue
        b = shingles(e.get("text", ""))
        if a and b:
            j = len(a & b) / len(a | b)
            if j > best:
                best, who = j, e.get("asin", "")
    return best, who


def videos_today(exclude_asin: str) -> int:
    today = dt.date.today().isoformat()
    return sum(1 for e in load_archive() if e.get("date") == today and e.get("asin") != exclude_asin)


# ----------------------------- LLM steps -----------------------------------
def _norm(t: str) -> str:
    """Replace look-alike hyphens/spaces that confuse fonts and voices."""
    for bad in ("\u2010", "\u2011", "\u2012", "\u2043"):
        t = t.replace(bad, "-")
    return t.replace("\u00a0", " ").replace("\u202f", " ")


def _clean_segments(raw: list[dict]) -> list[dict]:
    segs = []
    for i, s in enumerate(raw, 1):
        text = _norm(re.sub(r"\s+", " ", str(s.get("text", "")))).strip()
        if not text:
            continue
        segs.append({
            "id": str(s.get("id", i)), "section": str(s.get("section", "features")).strip().lower(),
            "text": text, "onscreen": str(s.get("onscreen", "") or "").strip(),
            "asset_ids": [str(a) for a in (s.get("asset_ids") or [])][:3],
            "source": str(s.get("source", "product")), "priority": int(s.get("priority", 2) or 2),
        })
    return segs


def salvage_segments(text: str) -> list[dict]:
    """Recover every COMPLETE segment object from a reply that was cut off or is otherwise malformed."""
    i = text.find('"segments"')
    i = text.find("[", i if i >= 0 else 0)
    if i < 0:
        return []
    dec, out, pos = json.JSONDecoder(), [], i + 1
    while True:
        j = text.find("{", pos)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(text[j:])
        except Exception:
            break  # the last object was cut off; keep what we have
        if isinstance(obj, dict) and obj.get("text"):
            out.append(obj)
        pos = j + end
    return out


def parse_script_reply(txt: str) -> dict | None:
    try:
        data = extract_json(txt)
    except Exception:
        data = None
    if isinstance(data, dict) and isinstance(data.get("segments"), list) and data["segments"]:
        return data
    if isinstance(data, list) and data and isinstance(data[0], dict) and "text" in data[0]:
        return {"segments": data}
    segs = salvage_segments(txt or "")
    return {"segments": segs, "salvaged": True} if len(segs) >= 8 else None


def generate(project: Project, cfg: dict, entries: list[dict], angle: str, target_words: int) -> dict:
    base = (f"ANGLE: {angle}: {ANGLES[angle]}\n\n{SECTION_GUIDE}\n\n"
            "LENGTH: about {words} words of narration in total across all segments (plus or minus 8%).\n\n"
            f"FACTS:\n{facts_yaml(project)}\n\nAVAILABLE FOOTAGE:\n{assets_block(entries)}\n\n"
            f"Return JSON in this shape:\n{SCHEMA}")
    words, note, data = target_words, "", None
    for attempt in range(3):
        txt = ask(cfg["llm"]["model"], SYSTEM, base.replace("{words}", str(words)) + note, max_tokens=12000, temperature=0.85)
        data = parse_script_reply(txt)
        if data:
            if data.get("salvaged"):
                log(f"  (reply was cut off; recovered {len(data['segments'])} complete lines)")
            break
        log(f"  script reply was not valid JSON (attempt {attempt + 1}); retrying with a shorter request")
        words = int(words * 0.8)
        note = ("\n\nIMPORTANT: your previous reply was cut off or was not valid JSON. Reply with ONLY one compact JSON object "
                "in the shape above. No commentary, no markdown, keep each narration line under 30 words.")
    if not data:
        raise RuntimeError("The model did not return a usable script after 3 tries. Try again, or choose another model in "
                           "config.yaml (llm.model). Reasoning models can run out of tokens; llama/maverick-style models are steadier.")
    segs = _clean_segments(data["segments"])
    return {"title": data.get("title") or project.facts["product_name"],
            "description_intro": data.get("description_intro", ""), "tags": data.get("tags", []),
            "angle": angle, "segments": segs}


FACTCHECK_SYSTEM = ("You are a strict fact-checker for product narration. Compare each narration segment against FACTS. "
                    "Answer ONLY with JSON.")


def factcheck(project: Project, cfg: dict, script: dict) -> list[dict]:
    # Your own fixed intro/outro lines are not product claims, so they are not fact-checked.
    segs = "\n".join(f"[{s['id']}] ({s['source']}) {s['text']}  || onscreen: {s['onscreen']}"
                     for s in script["segments"] if s.get("source") != "fixed")
    user = (f"FACTS:\n{facts_yaml(project)}\n\nSEGMENTS:\n{segs}\n\n"
            "Flag a segment ONLY if it: (a) states a product-specific detail not supported by FACTS; "
            "(b) states first-person experience not supported by hands_on_notes; "
            "(c) presents a manufacturer claim as the narrator's own finding; "
            "(d) makes a claim about durability, safety, performance or results that FACTS do not support; "
            "(e) puts unsupported text in onscreen. Do NOT flag generic category advice, transitions or opinions framed as 'if you...'.\n"
            'Return {"issues": [{"id": "<segment id>", "problem": "<short>", "fix": "<how to fix>"}]} - empty list if clean.')
    try:
        data = extract_json(ask(cfg["llm"]["model"], FACTCHECK_SYSTEM, user, max_tokens=3000, temperature=0.0))
        issues = data.get("issues", []) if isinstance(data, dict) else data
        return [i for i in issues if isinstance(i, dict)]
    except Exception as e:
        log(f"  fact-check reply unreadable ({e}); flagging for manual check")
        return [{"id": "*", "problem": "the automatic fact-check could not run (model reply unreadable)",
                 "fix": "check every line against your facts by hand"}]


def revise(project: Project, cfg: dict, script: dict, issues: list[dict]) -> dict:
    by_id = {s["id"]: s for s in script["segments"] if s.get("source") != "fixed"}
    todo = [i for i in issues if str(i.get("id")) in by_id]
    if not todo:
        return script
    lines = "\n".join(f"[{i['id']}] problem: {i['problem']} | fix: {i['fix']} | current: {by_id[str(i['id'])]['text']}" for i in todo)
    user = (f"FACTS:\n{facts_yaml(project)}\n\nRewrite ONLY these segments so they are fully supported by FACTS "
            f"(same voice, similar length). If a segment cannot be saved, return an empty text to delete it.\n{lines}\n\n"
            'Return {"segments": [{"id": "...", "text": "...", "onscreen": ""}]}')
    try:
        data = extract_json(ask(cfg["llm"]["model"], SYSTEM, user, max_tokens=4000, temperature=0.5))
        fixed = {str(s["id"]): s for s in (data.get("segments", []) if isinstance(data, dict) else []) if isinstance(s, dict)}
    except Exception as e:
        log(f"  revision reply unreadable ({e}); leaving those lines for you to fix")
        return script
    out = []
    for s in script["segments"]:
        f = fixed.get(s["id"]) if s.get("source") != "fixed" else None
        if f is None:
            out.append(s)
        elif str(f.get("text", "")).strip():
            s = dict(s, text=re.sub(r"\s+", " ", f["text"]).strip(), onscreen=str(f.get("onscreen", "") or "").strip())
            out.append(s)
    script["segments"] = out
    return script


def expand(project: Project, cfg: dict, entries: list[dict], script: dict, need_words: int) -> dict:
    """Add NEW lines to a script that is too short, without rewriting what is already there."""
    current = "\n".join(f"[{s['id']}] ({s['section']}) {s['text']}" for s in script["segments"])
    user = (f"FACTS:\n{facts_yaml(project)}\n\nAVAILABLE FOOTAGE:\n{assets_block(entries)}\n\nCURRENT SCRIPT:\n{current}\n\n"
            f"The script is too short for a 5 to 9 minute video. Write about {need_words} words of NEW segments to insert. "
            "Rules: never repeat anything already said. Add depth ONLY by (1) using facts from FACTS that are not used yet, "
            "(2) clearly hypothetical 'if you...' situations that claim nothing about this product beyond FACTS, "
            "(3) general advice about choosing or caring for this KIND of product (source = general; no claims about this specific product), "
            "(4) honest limits and who it does or does not suit. Do not invent specs, results or experiences. "
            "Never add a subscribe request, a sign-off or an outro line. "
            'Return {"segments": [{"after_id": "<id of the existing segment this follows>", "section": "...", "text": "...", '
            '"onscreen": "", "asset_ids": [], "source": "product|handson|general", "priority": 2}]}')
    txt = ask(cfg["llm"]["model"], SYSTEM, user, max_tokens=6500, temperature=0.8)
    data = parse_script_reply(txt)
    if not data:
        log("  expansion reply unreadable; keeping the script as it is")
        return script
    new = [dict(n, id="") for n in data["segments"] if isinstance(n, dict) and str(n.get("text", "")).strip()]
    segs = list(script["segments"])
    used = {s["id"] for s in segs}
    n_added = 0
    for n in new:
        after = str(n.get("after_id", ""))
        idx = next((i for i, s in enumerate(segs) if s["id"] == after), None)
        if idx is None:  # unknown anchor: put it at the end of the section it names, else before the outro
            sec = str(n.get("section", "")).lower()
            same = [i for i, s in enumerate(segs) if s["section"] == sec and s["section"] != "outro"]
            idx = same[-1] if same else None
        if idx is None or segs[idx]["section"] == "outro":
            first_outro = next((i for i, s in enumerate(segs) if s["section"] == "outro"), len(segs))
            idx = max(0, first_outro - 1)  # never insert inside or after the outro
        seg = _clean_segments([n])[0]
        seg["section"] = segs[idx]["section"]       # keep each section in one block
        k = 1
        while f"x{k}" in used:
            k += 1
        seg["id"] = f"x{k}"
        used.add(seg["id"])
        segs.insert(idx + 1, seg)
        n_added += 1
    log(f"  expansion added {n_added} line(s)")
    script["segments"] = segs
    return script


def trim_to_length(segments: list[dict], cfg: dict, max_minutes: float) -> list[dict]:
    segs = list(segments)
    for prio in (3, 2):
        while estimate_minutes(segs, cfg) > max_minutes:
            cand = [i for i, s in enumerate(segs)
                    if s["priority"] >= prio and s["section"] not in ("hook", "outro") and s.get("source") != "fixed"]
            if not cand:
                break
            del segs[cand[-1]]
    return segs


# ----------------------------- your intro and outro ------------------------
def _fixed(seg_id: str, section: str, text: str) -> dict:
    return {"id": seg_id, "section": section, "text": text, "onscreen": "", "asset_ids": [],
            "source": "fixed", "priority": 1}


def add_framing(script: dict, project: Project) -> dict:
    """Your own intro line (after the hook) and outro lines (at the end). No affiliate disclosure.

    Uses facts.yaml 'short_name' (e.g. 'cooling towels') when present, so the voice does not read the full product title.
    """
    name = project.facts.get("short_name") or project.facts.get("product_name") or "this product"
    segs = script["segments"]
    n_hook = 0
    for s in segs:  # count the leading run of hook segments
        if s["section"] != "hook":
            break
        n_hook += 1
    n_hook = max(1, min(n_hook, 3))

    intro = _fixed("I1", "hook",
                   f"Please subscribe to our channel if you haven't yet. Let's get started with the {name}.")
    outro = [
        _fixed("O1", "outro",
               f"If you liked this video, give it a like and subscribe to the channel. "
               f"In the comments, tell me if you liked the {name}."),
        _fixed("O2", "outro", "I've put the product links in the video description below."),
        _fixed("O3", "outro",
               "Thank you very much for watching the whole video. Hope to see you in the next video. Take care."),
    ]
    script["segments"] = segs[:n_hook] + [intro] + segs[n_hook:] + outro
    return script


# ----------------------------- orchestration -------------------------------
def make_script(project: Project, cfg: dict, entries: list[dict], angle: str | None = None, force: bool = False) -> dict:
    sc = cfg["safety"]
    if not force and videos_today(project.asin) >= sc["max_videos_per_day"]:
        raise SystemExit(f"Daily limit of {sc['max_videos_per_day']} videos reached (anti mass-production brake). Use --force to override.")
    ln = cfg["length"]
    angle = pick_angle(project, cfg, angle)
    fixed_words = 80  # your intro line plus the three outro lines
    lo, hi = ln["min_minutes"] + 0.35, ln["max_minutes"] - 0.45
    target_min = min(max(ln["target_minutes"], lo + 0.3), hi - 0.3)
    target_words = int((target_min * 60 - ln["lead_in"] - ln["tail"]) * ln["wpm"] / 60 * 0.9) - fixed_words
    log(f"Angle: {angle}; target ~{target_words} words")

    script = generate(project, cfg, entries, angle, target_words)
    script = add_framing(script, project)
    est = estimate_minutes(assemble(script, project, cfg), cfg)
    log(f"  first draft: ~{est:.1f} min")
    if est > hi:
        overhead = est - estimate_minutes(script["segments"], cfg)
        script["segments"] = trim_to_length(script["segments"], cfg, hi - overhead)
        est = estimate_minutes(assemble(script, project, cfg), cfg)
    for rnd in range(4):
        if est >= lo:
            break
        need = int((target_min - est) * 60 * ln["wpm"] / 60 * 0.85)
        need = max(120, min(need, 650))
        log(f"  too short ({est:.1f} min); expanding by about {need} words (round {rnd + 1})")
        script = expand(project, cfg, entries, script, need)
        est = estimate_minutes(assemble(script, project, cfg), cfg)
    if est > hi:
        overhead = est - estimate_minutes(script["segments"], cfg)
        script["segments"] = trim_to_length(script["segments"], cfg, hi - overhead)
        est = estimate_minutes(assemble(script, project, cfg), cfg)
    log(f"  final length ~{est:.1f} min")

    review: dict = {"estimated_minutes": round(est, 1), "fact_issues": [], "policy": [], "style": []}
    if est < ln["min_minutes"]:
        words_needed = int((ln["min_minutes"] + 0.2 - est) * 60 * ln["wpm"] / 60)
        review["length_warning"] = (f"Script is only ~{est:.1f} min; the video must be at least {ln['min_minutes']:.0f}. "
                                    f"Add about {words_needed} more words (roughly {words_needed // 22} lines), "
                                    "ideally by adding real facts and hands_on_notes to facts.yaml and re-running the script step with force.")
    for rnd in range(2):
        issues = factcheck(project, cfg, script)
        log(f"  fact-check round {rnd + 1}: {len(issues)} issue(s)")
        review["fact_issues"] = issues
        if not issues or all(i.get("id") == "*" for i in issues):
            break
        script = revise(project, cfg, script, issues)
    if review["fact_issues"]:
        review["fact_issues"] = factcheck(project, cfg, script)  # what is still open after revision

    for s in script["segments"]:
        pol, sty = check_text(s["text"] + " " + s["onscreen"])
        if pol:
            review["policy"].append({"id": s["id"], "problems": pol})
        if sty:
            review["style"].append({"id": s["id"], "problems": sty})

    body = " ".join(s["text"] for s in script["segments"])
    sim, who = max_similarity(body, project.asin)
    review["similarity"] = round(sim, 3)
    if sim > sc["max_similarity"]:
        review["similarity_note"] = f"{sim:.0%} 4-gram overlap with the script for {who}. Rewrite before posting."
    script["review"] = review
    return script


# ----------------------------- final segment list ---------------------------
def assemble(script: dict, project: Project, cfg: dict) -> list[dict]:
    """The segments to render. The intro and outro are already part of the script; no disclosure is added."""
    segs = [dict(s) for s in script["segments"]]
    for s in segs:
        s.pop("endcard", None)
    if segs and segs[-1]["section"] == "outro":
        segs[-1]["endcard"] = True
    return segs


# ----------------------------- markdown round-trip --------------------------
def script_to_md(script: dict, project: Project) -> str:
    L = [f"# TITLE: {script.get('title', '')}", f"# INTRO: {script.get('description_intro', '')}",
         f"# TAGS: {', '.join(script.get('tags', []))}", f"# ANGLE: {script.get('angle', '')}",
         "#", "# HOW TO EDIT: change any wording, delete a line to cut it, add a line with a new id (e.g. [n1]).",
         "# Format:  [id] narration | callout: up to 4 words | assets: A01,A02 | p: 1-3",
         "# Intro and outro lines are part of this script (ids I1, O1, O2, O3). No affiliate disclosure is added; "
         "put it in the description and pinned comment. Do not add prices, ratings or review quotes.",
         "# Edit this file in place (projects/%s/script.md), commit, then run the render step." % project.asin, ""]
    cur = None
    for s in script["segments"]:
        if s["section"] != cur:
            cur = s["section"]
            L += ["", f"## {cur}"]
        text = s["text"].replace(" | ", ", ")
        opts = []
        if s.get("onscreen"):
            opts.append(f"callout: {s['onscreen']}")
        if s.get("asset_ids"):
            opts.append("assets: " + ",".join(s["asset_ids"]))
        opts.append(f"p: {s.get('priority', 2)}")
        L.append(f"[{s['id']}] {text} | " + " | ".join(opts))
    rv = script.get("review") or {}
    if rv:
        L += ["", "<!-- REVIEW NOTES (ignored when rendering)"]
        L.append(f"estimated length: {rv.get('estimated_minutes')} min; similarity to earlier scripts: {rv.get('similarity')}")
        if rv.get("length_warning"):
            L.append("LENGTH: " + rv["length_warning"])
        for i in rv.get("fact_issues", []):
            L.append(f"FACT CHECK [{i.get('id')}]: {i.get('problem')} -> {i.get('fix')}")
        for i in rv.get("policy", []):
            L.append(f"POLICY [{i['id']}]: {', '.join(i['problems'])}")
        for i in rv.get("style", []):
            L.append(f"STYLE [{i['id']}]: {', '.join(i['problems'])}")
        if rv.get("similarity_note"):
            L.append(rv["similarity_note"])
        L.append("-->")
    return "\n".join(L) + "\n"


def md_to_script(md: str) -> dict:
    meta = {"title": "", "description_intro": "", "tags": [], "angle": ""}
    segs, section = [], "features"
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    for raw in md.splitlines():
        line = raw.strip()
        if line.startswith("# TITLE:"):
            meta["title"] = line[8:].strip()
        elif line.startswith("# INTRO:"):
            meta["description_intro"] = line[8:].strip()
        elif line.startswith("# TAGS:"):
            meta["tags"] = [t.strip() for t in line[7:].split(",") if t.strip()]
        elif line.startswith("# ANGLE:"):
            meta["angle"] = line[8:].strip()
        elif line.startswith("## "):
            section = line[3:].strip().lower()
        else:
            m = re.match(r"^\[(\w+)\]\s*(.+)$", line)
            if not m:
                continue
            parts = [p.strip() for p in m.group(2).split(" | ")]
            seg = {"id": m.group(1), "section": section, "text": _norm(parts[0]), "onscreen": "", "asset_ids": [],
                   "source": "product", "priority": 2}
            for p in parts[1:]:
                if p.lower().startswith("callout:"):
                    seg["onscreen"] = p[8:].strip()
                elif p.lower().startswith("assets:"):
                    seg["asset_ids"] = [a.strip() for a in p[7:].split(",") if a.strip()]
                elif p.lower().startswith("p:"):
                    try:
                        seg["priority"] = int(p[2:].strip())
                    except ValueError:
                        pass
            segs.append(seg)
    meta["segments"] = segs
    return meta
