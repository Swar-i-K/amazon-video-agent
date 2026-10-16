"""Write a natural, fact-bound narration script, check it, and keep it within 5-9 minutes."""
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
    "comparison": "A fair side-by-side comparison of 2-3 products for one kind of buyer (used automatically for comparison projects).",
    "list": "A Top-N list of 3-9 products ranked by the creator, each presented as the creator's pick with honest limits (used automatically for list projects).",
    "first_impressions": "A hands-on first look: what it is, what stands out, and what a buyer should know before deciding.",
    "who_its_for": "A decision guide: who this suits, who should skip it, and how to decide for yourself.",
    "problem_solution": "Open with the everyday problem this product addresses, then walk through how each feature deals with it.",
    "buyers_checklist": "Teach the viewer what to check when choosing this kind of product, then measure this one against that checklist.",
    "day_in_the_life": "Walk through realistic situations where it gets used, tying each moment to a supplied fact.",
    "honest_pros_cons": "A balanced rundown that takes the downsides as seriously as the upsides.",
    "gift_angle": "Frame it around who it would suit as a gift and what to consider before gifting it.",
}

SECTION_LABELS = {
    "head_to_head": "Head to head", "criteria": "How I picked", "recap": "Recap",
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
7. Do not write a disclosure line or a sign-off; both are added automatically.
8. Write for the ear. Each segment is 1-3 sentences, 14-32 words. Spell out units ("20 ounces"). No emojis, no markdown, no abbreviations a voice would mispronounce.
9. onscreen is optional: at most 4 words, and only a fact that appears in FACTS (e.g. "20 oz capacity"). Otherwise "".
10. asset_ids: pick 1-3 ids from AVAILABLE FOOTAGE that match what the narration is talking about. Use [] if nothing fits. Never describe something the footage does not show.
11. KEYWORDS, if present in FACTS, are phrases viewers search for. Work 2 to 4 of them naturally into the title and description_intro (the title should contain the first keyword). Never stuff or repeat them, and never let a keyword imply a claim FACTS do not support.
12. Output ONLY JSON, no commentary."""

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


COMPARE_SYSTEM = """You write narration for a faceless COMPARISON video of 2 or 3 products. A human creator reviews every script, an AI voice reads it, and the creator posts it.
Your narration must sound like one real person talking to one viewer: contractions, varied sentence length (some very short), concrete detail, natural transitions, honest asides. No sales-pitch voice.

HARD RULES
1. Facts come ONLY from FACTS, and every fact belongs to exactly ONE product. Say the product's label whenever you state a product-specific detail. NEVER move or blend a fact between products.
2. Compare a point ONLY if EVERY product being compared has a fact for it. If one product lacks it, skip the point, or say you are only describing the one that has it. Never guess the missing side.
3. manufacturer_claims are the brand's words: attribute them ("the brand says").
4. First-person experience only from that product's own hands_on_notes. A personal preference ("I'd pick X for ...") is allowed ONLY when PICK_ALLOWED is true and it is tied to a stated observation. When PICK_ALLOWED is false, do not pick a winner: say which kind of buyer each product suits, using the facts.
5. Words like better, best, worse, wins, beats, outperforms are allowed ONLY for a point where the hands_on_notes of BOTH products support the difference.
6. General buying advice is allowed (source = general) if it claims nothing specific about any of the products.
7. NEVER mention: prices, discounts, deals, coupons, star ratings, review counts, what customers say, "best seller", "Amazon's Choice", other retailers, health claims, guarantees.
8. Avoid: game-changer, revolutionary, must-have, life-changing, perfect for everyone, "in today's video". No hype.
9. Write for the ear. Each segment is 1-3 sentences, 14-32 words. Spell out units. No emojis, markdown or abbreviations a voice would mispronounce.
10. onscreen is optional: at most 4 words, only a fact from FACTS (e.g. "Pack of 4"). Otherwise "".
11. product = "1", "2" or "3" for a line about ONE product; "all" for lines about the comparison itself or general advice. asset_ids: 1-3 ids from AVAILABLE FOOTAGE that belong to the product the line is about (ids start with P1-, P2-, P3-; X- is shared footage). For "all" lines any footage is fine. Never describe something the footage does not show.
12. Do not write a disclosure line or a sign-off; both are added automatically.
13. KEYWORDS, if present in FACTS, are phrases viewers search for. Work 2 to 4 of them naturally into the title and description_intro (the title should contain the first keyword). Never stuff or repeat them, and never let a keyword imply a claim FACTS do not support.
14. Output ONLY JSON, no commentary."""

COMPARE_GUIDE = (
    "Section order: hook (2-3 lines: the question a buyer is asking, no greeting), context (what to look for in this kind of product), "
    "overview (one line per product: its label and what it is), product_1, product_2 and, if there is a third, product_3 "
    "(each: key facts, hands-on notes, honest limits; 4-8 lines; use exactly these section names), head_to_head "
    "(same-point comparisons, ONLY where every product has facts; 6-12 lines), who_for (which kind of buyer each product suits), "
    "verdict (honest summary that follows rule 4)."
)

COMPARE_SCHEMA = """{
 "title": "YouTube title, under 70 chars, e.g. 'A vs B: cooling towels compared'; no clickbait, no price",
 "description_intro": "2-3 plain sentences describing the comparison, no hype, no price",
 "tags": ["8-12 relevant tags"],
 "segments": [
  {"id": 1, "section": "hook|context|overview|product_1|product_2|product_3|head_to_head|who_for|verdict",
   "product": "1|2|3|all", "text": "narration", "onscreen": "", "asset_ids": ["P1-A01"], "source": "product|handson|general", "priority": 1}
 ]
}
priority: 1 = essential, 2 = useful, 3 = nice to have (cut first if the video runs long)."""


LIST_SYSTEM = """You write narration for a faceless TOP-LIST video ("Top N ...") that presents 3 to 9 products the creator chose and ranked. A human creator reviews every script, an AI voice reads it, and the creator posts it.
Your narration must sound like one real person talking to one viewer: contractions, varied sentence length (some very short), concrete detail, natural transitions, honest asides. No sales-pitch voice.

HARD RULES
1. Facts come ONLY from FACTS, and every fact belongs to exactly ONE product. Say the product's label whenever you state a product-specific detail. NEVER move or blend a fact between products.
2. The ORDER of the list, each product's rank_label and best_for, and the CRITERIA are the creator's own decisions. Present them as "my pick for ...", never as an objective or tested result. Explain a ranking ONLY with that product's facts, its hands_on_notes and the CRITERIA. Never invent test results, sales or popularity rankings, or comparisons with products that are not on the list.
3. If a product has fewer than 3 hands_on_notes, do not claim you tested it in any particular way: describe it from its facts ("on paper", "the brand says") and say plainly that the ranking follows the stated criteria.
4. manufacturer_claims are the brand's words: attribute them ("the brand says").
5. First-person experience only from that product's own hands_on_notes.
6. Each item has: what it is (one line), why it sits at this rank (facts, notes, criteria), who it suits (best_for), and one honest limit (cons / not_for) when given.
7. NEVER mention: prices, price level (cheap, affordable, budget), discounts, deals, coupons, star ratings, review counts, what customers say, "best seller", "Amazon's Choice", other retailers, health claims, guarantees.
8. Avoid: game-changer, revolutionary, must-have, life-changing, perfect for everyone, "in today's video". No hype.
9. Write for the ear. Each segment is 1-3 sentences, 14-32 words. Spell out units. No emojis, markdown or abbreviations a voice would mispronounce.
10. onscreen is optional: at most 4 words, only a fact from FACTS (e.g. "Pack of 4"). Otherwise "".
11. product = the item's RANK NUMBER as a string ("1" to "9") for a line about that one product; "all" for the intro, criteria, recap and verdict lines. asset_ids: 1-3 ids from AVAILABLE FOOTAGE that belong to the product the line is about (ids start with P1-, P2- ... ; X- is shared footage). Never describe something the footage does not show.
12. Do not write a disclosure line or a sign-off; both are added automatically.
13. KEYWORDS, if present in FACTS, are phrases viewers search for. Work 2 to 4 of them naturally into the title and description_intro (the title should contain the first keyword). Never stuff or repeat them, and never let a keyword imply a claim FACTS do not support.
14. Output ONLY JSON, no commentary."""

LIST_GUIDE = (
    "Section order: hook (2-3 lines: the question a buyer is asking, no greeting), criteria (how the list was chosen and ranked: use the "
    "creator's CRITERIA only; 2-4 lines), then ONE section per product named item_<rank> (for example item_6, item_5 ... item_1), in EXACTLY "
    "the presentation order given by ORDER in FACTS, 5-9 lines each, then recap (one line per product: rank, label, who it suits; in rank order), "
    "verdict (who should start with which pick; no unsupported 'best')."
)

LIST_SCHEMA = """{
 "title": "YouTube title, under 70 chars, e.g. 'Top 6 cooling towels for hot weather'; contains the first keyword; no clickbait, no price",
 "description_intro": "2-3 plain sentences describing the list, no hype, no price",
 "tags": ["8-12 relevant tags"],
 "segments": [
  {"id": 1, "section": "hook|criteria|item_<rank>|recap|verdict",
   "product": "1-9|all", "text": "narration", "onscreen": "", "asset_ids": ["P1-A01"], "source": "product|handson|general", "priority": 1}
 ]
}
priority: 1 = essential, 2 = useful, 3 = nice to have (cut first if the video runs long)."""


def _kind(project: Project):
    return project.compare["kind"] if project.compare else None


def _system(project: Project) -> str:
    k = _kind(project)
    return LIST_SYSTEM if k == "list" else COMPARE_SYSTEM if k == "compare" else SYSTEM


def _guide(project: Project) -> str:
    k = _kind(project)
    return LIST_GUIDE if k == "list" else COMPARE_GUIDE if k == "compare" else SECTION_GUIDE


def _schema(project: Project) -> str:
    k = _kind(project)
    return LIST_SCHEMA if k == "list" else COMPARE_SCHEMA if k == "compare" else SCHEMA


# ---------------------------------------------------------------------------
_FACT_KEYS = ["product_name", "brand", "category", "audience", "keywords", "manufacturer_claims", "specs",
              "hands_on_notes", "pros", "cons", "not_for", "use_cases", "notes"]


def _facts_one(f: dict) -> str:
    d = {k: f[k] for k in _FACT_KEYS if f.get(k)}
    return yaml.safe_dump(d, sort_keys=False, allow_unicode=True, width=1000)


def facts_yaml(project: Project) -> str:
    if project.compare:
        c = project.compare
        parts = []
        if c["kind"] == "list":
            n = len(c["products"])
            order = list(range(n, 0, -1)) if c["order"] == "countdown" else list(range(1, n + 1))
            parts.append(f"LIST: {project.facts['product_name']}\nORDER: {c['order']} (present the sections in this order: "
                         + ", ".join(f"item_{k}" for k in order) + ")\nCRITERIA: " + (c["criteria"] or "(none given: say only that this is your order)"))
        for i, (sub, label) in enumerate(zip(c["products"], c["labels"]), 1):
            head = f"PRODUCT {i}  (" + (f"rank #{i}; " if c["kind"] == "list" else "") + f"label: {label}; folder: {sub.asin}"
            if c["kind"] == "list":
                if c["rank_labels"][i - 1]:
                    head += f"; rank_label: {c['rank_labels'][i - 1]}"
                if c["best_for"][i - 1]:
                    head += f"; best_for: {c['best_for'][i - 1]}"
                head += f"; hands_on_notes: {len(sub.facts.get('hands_on_notes') or [])}"
            parts.append(head + f")\n{_facts_one(sub.facts)}")
        extra = {k: project.facts[k] for k in ("compare_points", "notes") if project.facts.get(k)}
        if extra:
            parts.append("COMPARISON NOTES\n" + yaml.safe_dump(extra, sort_keys=False, allow_unicode=True, width=1000))
        if project.facts.get("keywords"):
            parts.append("KEYWORDS: " + ", ".join(project.facts["keywords"]))
        parts.append(f"PICK_ALLOWED: {'true' if c['pick_ok'] else 'false'}")
        return "\n\n".join(parts)
    return _facts_one(project.facts)


def assets_block(entries: list[dict]) -> str:
    lines = []
    for e in entries:
        if e["usable"]:
            extra = f", {e['duration']:.0f}s" if e["kind"] == "clip" else ""
            who = f" ({e['label']})" if e.get("label") else ""
            lines.append(f"{e['id']} [{e['kind']}{extra}]{who} {e['description']}")
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
    if project.compare:
        return "list" if project.compare["kind"] == "list" else "comparison"
    if forced:
        if forced not in ANGLES or forced in ("comparison", "list"):
            raise SystemExit(f"Unknown angle '{forced}'. Choose from: {', '.join(a for a in ANGLES if a not in ('comparison', 'list'))}")
        return forced
    if project.facts.get("angle") in ANGLES and project.facts.get("angle") not in ("comparison", "list"):
        return project.facts["angle"]
    recent = [e.get("angle") for e in load_archive()[-8:]]
    pool = sorted((a for a in ANGLES if a not in ("comparison", "list")), key=lambda a: (recent.count(a), a))
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


def _product_tag(v) -> str:
    t = str(v or "all").strip().lower()
    m = re.search(r"[1-9]", t)
    return m.group(0) if m and t != "all" else "all"


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
            "product": _product_tag(s.get("product")),
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
    base = (f"ANGLE: {angle}: {ANGLES[angle]}\n\n{_guide(project)}\n\n"
            "LENGTH: about {words} words of narration in total across all segments (plus or minus 8%).\n\n"
            f"FACTS:\n{facts_yaml(project)}\n\nAVAILABLE FOOTAGE:\n{assets_block(entries)}\n\n"
            f"Return JSON in this shape:\n{_schema(project)}")
    words, note, data = target_words, "", None
    for attempt in range(3):
        txt = ask(cfg["llm"]["model"], _system(project), base.replace("{words}", str(words)) + note, max_tokens=12000, temperature=0.85)
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
    kws = list(project.facts.get("keywords") or [])
    tags = []
    for t in kws + [str(x) for x in (data.get("tags") or [])]:
        if t.strip() and t.strip().lower() not in {x.lower() for x in tags}:
            tags.append(t.strip())
    return {"title": data.get("title") or project.facts["product_name"],
            "description_intro": data.get("description_intro", ""), "tags": tags[:15],
            "angle": angle, "segments": segs}


FACTCHECK_SYSTEM = ("You are a strict fact-checker for product narration. Compare each narration segment against FACTS. "
                    "Answer ONLY with JSON.")


def factcheck(project: Project, cfg: dict, script: dict) -> list[dict]:
    tag = (lambda s: f" [product {s.get('product', 'all')}]") if project.compare else (lambda s: "")
    segs = "\n".join(f"[{s['id']}] ({s['source']}){tag(s)} {s['text']}  || onscreen: {s['onscreen']}" for s in script["segments"])
    extra = ""
    if _kind(project) == "list":
        extra = ("(f) attributes a fact, hands-on observation or manufacturer claim to the WRONG product, or blends two products; "
                 "(g) states or implies a test, measurement or comparison the creator did not make (the ranking is the creator's opinion and "
                 "must be presented as such, supported only by FACTS and CRITERIA); (h) invents ranking criteria that are not in CRITERIA; "
                 "(i) calls a product the best / number one / top other than as the creator's own pick for its rank_label or best_for. ")
    elif project.compare:
        extra = ("(f) attributes a fact, hands-on observation or manufacturer claim to the WRONG product, or blends two products; "
                 "(g) compares a point when any compared product has no fact for it; "
                 "(h) declares a winner or says better/worse/best without the hands_on_notes of BOTH products supporting it "
                 "(or at all when PICK_ALLOWED is false). ")
    user = (f"FACTS:\n{facts_yaml(project)}\n\nSEGMENTS:\n{segs}\n\n"
            "Flag a segment ONLY if it: (a) states a product-specific detail not supported by FACTS; "
            "(b) states first-person experience not supported by hands_on_notes; "
            "(c) presents a manufacturer claim as the narrator's own finding; "
            "(d) makes a claim about durability, safety, performance or results that FACTS do not support; "
            "(e) puts unsupported text in onscreen. " + extra + "Do NOT flag generic category advice, transitions or opinions framed as 'if you...'.\n"
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
    by_id = {s["id"]: s for s in script["segments"]}
    todo = [i for i in issues if str(i.get("id")) in by_id]
    if not todo:
        return script
    lines = "\n".join(f"[{i['id']}] problem: {i['problem']} | fix: {i['fix']} | current: {by_id[str(i['id'])]['text']}" for i in todo)
    user = (f"FACTS:\n{facts_yaml(project)}\n\nRewrite ONLY these segments so they are fully supported by FACTS "
            f"(same voice, similar length). If a segment cannot be saved, return an empty text to delete it.\n{lines}\n\n"
            'Return {"segments": [{"id": "...", "text": "...", "onscreen": ""}]}')
    try:
        data = extract_json(ask(cfg["llm"]["model"], _system(project), user, max_tokens=4000, temperature=0.5))
        fixed = {str(s["id"]): s for s in (data.get("segments", []) if isinstance(data, dict) else []) if isinstance(s, dict)}
    except Exception as e:
        log(f"  revision reply unreadable ({e}); leaving those lines for you to fix")
        return script
    out = []
    for s in script["segments"]:
        f = fixed.get(s["id"])
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
            'Return {"segments": [{"after_id": "<id of the existing segment this follows>", "section": "...", "text": "...", '
            '"onscreen": "", "asset_ids": [], "source": "product|handson|general", "priority": 2'
            + ((', "product": "1-9|all"' if _kind(project) == "list" else ', "product": "1|2|3|all"') if project.compare else "") + "}]}"
            + (" Keep every fact with its own product, and never compare a point unless every compared product has a fact for it."
               if project.compare else ""))
    txt = ask(cfg["llm"]["model"], _system(project), user, max_tokens=6500, temperature=0.8)
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
        if idx is None:  # unknown anchor: put it at the end of the section it names, else before the last section
            sec = str(n.get("section", "")).lower()
            same = [i for i, s in enumerate(segs) if s["section"] == sec]
            idx = same[-1] if same else max(0, len(segs) - 2)
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
            cand = [i for i, s in enumerate(segs) if s["priority"] >= prio and s["section"] != "hook"]
            if not cand:
                break
            del segs[cand[-1]]
    return segs


# ----------------------------- orchestration -------------------------------
def make_script(project: Project, cfg: dict, entries: list[dict], angle: str | None = None, force: bool = False) -> dict:
    sc = cfg["safety"]
    if not force and videos_today(project.asin) >= sc["max_videos_per_day"]:
        raise SystemExit(f"Daily limit of {sc['max_videos_per_day']} videos reached (anti mass-production brake). Use --force to override.")
    ln = cfg["length"]
    angle = pick_angle(project, cfg, angle)
    fixed_words = 45
    lo, hi = ln["min_minutes"] + 0.35, ln["max_minutes"] - 0.45
    target_min = min(max(ln["target_minutes"], lo + 0.3), hi - 0.3)
    target_words = int((target_min * 60 - ln["lead_in"] - ln["tail"]) * ln["wpm"] / 60 * 0.9) - fixed_words
    log(f"Angle: {angle}; target ~{target_words} words")

    script = generate(project, cfg, entries, angle, target_words)
    est = estimate_minutes(assemble(script, project, cfg), cfg)
    log(f"  first draft: ~{est:.1f} min")
    if est > hi:
        overhead = est - estimate_minutes(script["segments"], cfg)  # disclosure + sign-off are added later
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


# ----------------------------- fixed segments ------------------------------
def assemble(script: dict, project: Project, cfg: dict) -> list[dict]:
    """Body segments + automatically inserted disclosure + sign-off."""
    rng = random.Random(project.asin)
    d, segs = cfg["disclosure"], script["segments"]
    n_hook = 0
    for s in segs:  # count the leading run of hook segments
        if s["section"] != "hook":
            break
        n_hook += 1
    n_hook = max(1, min(n_hook, 3))
    disclosure = {"id": "D", "section": "disclosure", "text": rng.choice(d["spoken"]), "onscreen": "",
                  "asset_ids": [], "source": "fixed", "priority": 1, "banner": True, "product": "all"}
    k = _kind(project)
    outro_pool = (d.get("outro_list") if k == "list" else d.get("outro_compare") if k == "compare" else None) or d["outro"]
    outro = {"id": "O", "section": "outro", "text": rng.choice(outro_pool), "onscreen": "",
             "asset_ids": [], "source": "fixed", "priority": 1, "endcard": True, "product": "all"}
    if d.get("mode", "full") != "full":      # no spoken/banner disclosure inside the video
        return segs + [outro]
    return segs[:n_hook] + [disclosure] + segs[n_hook:] + [outro]


# ----------------------------- markdown round-trip --------------------------
def script_to_md(script: dict, project: Project) -> str:
    L = [f"# TITLE: {script.get('title', '')}", f"# INTRO: {script.get('description_intro', '')}",
         f"# TAGS: {', '.join(script.get('tags', []))}", f"# ANGLE: {script.get('angle', '')}"]
    if project.compare:
        L.append("# PRODUCTS: " + "; ".join(f"{i}={lab} ({sub.asin})" for i, (sub, lab) in
                                            enumerate(zip(project.compare["products"], project.compare["labels"]), 1)))
        if project.compare["kind"] == "list":
            L.append(f"# ORDER: {project.compare['order']} (1 = your best pick; sections run in presentation order)")
    L += ["#", "# HOW TO EDIT: change any wording, delete a line to cut it, add a line with a new id (e.g. [n1]).",
         "# Format:  [id] narration | product: 1-9/all | callout: up to 4 words | assets: A01,A02 | p: 1-3",
         "# The sign-off is added automatically; where the affiliate disclosure appears is set in config.yaml (disclosure: mode). Do not add prices, ratings or review quotes.",
         "# Edit this file in place (projects/%s/script.md), commit, then run the render step." % project.asin, ""]
    cur = None
    for s in script["segments"]:
        if s["section"] != cur:
            cur = s["section"]
            L += ["", f"## {cur}"]
        text = s["text"].replace(" | ", ", ")
        opts = []
        if s.get("product") and s["product"] != "all":
            opts.append(f"product: {s['product']}")
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
                   "source": "product", "priority": 2, "product": "all"}
            for p in parts[1:]:
                if p.lower().startswith("product:"):
                    seg["product"] = _product_tag(p[8:])
                elif p.lower().startswith("callout:"):
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
