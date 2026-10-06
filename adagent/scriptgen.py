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
7. Do not write a disclosure line or a sign-off; both are added automatically.
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
def _clean_segments(raw: list[dict]) -> list[dict]:
    segs = []
    for i, s in enumerate(raw, 1):
        text = re.sub(r"\s+", " ", str(s.get("text", ""))).strip()
        if not text:
            continue
        segs.append({
            "id": str(s.get("id", i)), "section": str(s.get("section", "features")).strip().lower(),
            "text": text, "onscreen": str(s.get("onscreen", "") or "").strip(),
            "asset_ids": [str(a) for a in (s.get("asset_ids") or [])][:3],
            "source": str(s.get("source", "product")), "priority": int(s.get("priority", 2) or 2),
        })
    return segs


def generate(project: Project, cfg: dict, entries: list[dict], angle: str, target_words: int) -> dict:
    user = (f"ANGLE: {angle}: {ANGLES[angle]}\n\n{SECTION_GUIDE}\n\n"
            f"LENGTH: about {target_words} words of narration in total across all segments (plus or minus 8%).\n\n"
            f"FACTS:\n{facts_yaml(project)}\n\nAVAILABLE FOOTAGE:\n{assets_block(entries)}\n\n"
            f"Return JSON in this shape:\n{SCHEMA}")
    data = extract_json(ask(cfg["llm"]["model"], SYSTEM, user, max_tokens=12000, temperature=0.85))
    segs = _clean_segments(data["segments"])
    return {"title": data.get("title", project.facts["product_name"]),
            "description_intro": data.get("description_intro", ""), "tags": data.get("tags", []),
            "angle": angle, "segments": segs}


FACTCHECK_SYSTEM = ("You are a strict fact-checker for product narration. Compare each narration segment against FACTS. "
                    "Answer ONLY with JSON.")


def factcheck(project: Project, cfg: dict, script: dict) -> list[dict]:
    segs = "\n".join(f"[{s['id']}] ({s['source']}) {s['text']}  || onscreen: {s['onscreen']}" for s in script["segments"])
    user = (f"FACTS:\n{facts_yaml(project)}\n\nSEGMENTS:\n{segs}\n\n"
            "Flag a segment ONLY if it: (a) states a product-specific detail not supported by FACTS; "
            "(b) states first-person experience not supported by hands_on_notes; "
            "(c) presents a manufacturer claim as the narrator's own finding; "
            "(d) makes a claim about durability, safety, performance or results that FACTS do not support; "
            "(e) puts unsupported text in onscreen. Do NOT flag generic category advice, transitions or opinions framed as 'if you...'.\n"
            'Return {"issues": [{"id": "<segment id>", "problem": "<short>", "fix": "<how to fix>"}]} - empty list if clean.')
    data = extract_json(ask(cfg["llm"]["model"], FACTCHECK_SYSTEM, user, max_tokens=3000, temperature=0.0))
    return list(data.get("issues", []))


def revise(project: Project, cfg: dict, script: dict, issues: list[dict]) -> dict:
    by_id = {s["id"]: s for s in script["segments"]}
    todo = [i for i in issues if str(i.get("id")) in by_id]
    if not todo:
        return script
    lines = "\n".join(f"[{i['id']}] problem: {i['problem']} | fix: {i['fix']} | current: {by_id[str(i['id'])]['text']}" for i in todo)
    user = (f"FACTS:\n{facts_yaml(project)}\n\nRewrite ONLY these segments so they are fully supported by FACTS "
            f"(same voice, similar length). If a segment cannot be saved, return an empty text to delete it.\n{lines}\n\n"
            'Return {"segments": [{"id": "...", "text": "...", "onscreen": ""}]}')
    data = extract_json(ask(cfg["llm"]["model"], SYSTEM, user, max_tokens=4000, temperature=0.5))
    fixed = {str(s["id"]): s for s in data.get("segments", [])}
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

    script = None
    for attempt in range(3):
        script = generate(project, cfg, entries, angle, target_words)
        est = estimate_minutes(assemble(script, project, cfg), cfg)
        log(f"  draft {attempt + 1}: ~{est:.1f} min")
        if est > hi:
            overhead = est - estimate_minutes(script["segments"], cfg)  # disclosure + sign-off are added later
            script["segments"] = trim_to_length(script["segments"], cfg, hi - overhead)
            est = estimate_minutes(assemble(script, project, cfg), cfg)
        if lo <= est <= hi:
            break
        if est < lo:
            target_words = int(target_words * (target_min / max(est, 1.0)) * 1.05)
    est = estimate_minutes(assemble(script, project, cfg), cfg)

    review: dict = {"estimated_minutes": round(est, 1), "fact_issues": [], "policy": [], "style": []}
    for rnd in range(2):
        issues = factcheck(project, cfg, script)
        log(f"  fact-check round {rnd + 1}: {len(issues)} issue(s)")
        review["fact_issues"] = issues
        if not issues:
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
                  "asset_ids": [], "source": "fixed", "priority": 1, "banner": True}
    outro = {"id": "O", "section": "outro", "text": rng.choice(d["outro"]), "onscreen": "",
             "asset_ids": [], "source": "fixed", "priority": 1, "endcard": True}
    return segs[:n_hook] + [disclosure] + segs[n_hook:] + [outro]


# ----------------------------- markdown round-trip --------------------------
def script_to_md(script: dict, project: Project) -> str:
    L = [f"# TITLE: {script.get('title', '')}", f"# INTRO: {script.get('description_intro', '')}",
         f"# TAGS: {', '.join(script.get('tags', []))}", f"# ANGLE: {script.get('angle', '')}",
         "#", "# HOW TO EDIT: change any wording, delete a line to cut it, add a line with a new id (e.g. [n1]).",
         "# Format:  [id] narration | callout: up to 4 words | assets: A01,A02 | p: 1-3",
         "# The affiliate disclosure and the sign-off are added automatically. Do not add prices, ratings or review quotes.",
         "# Save as  projects/%s/script_approved.md  and run the render step." % project.asin, ""]
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
            seg = {"id": m.group(1), "section": section, "text": parts[0], "onscreen": "", "asset_ids": [],
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
