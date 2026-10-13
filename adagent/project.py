"""Project folder handling, facts validation, and hard content rules."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .common import ROOT

ASIN_RE = re.compile(r"^[A-Z0-9][A-Z0-9_-]{2,39}$")   # a product ASIN, or the name of a comparison folder
AMAZON_LINK_RE = re.compile(
    r"^https://((www\.)?amazon\.[a-z.]{2,6}/|amzn\.(to|eu|asia)/|a\.co/)", re.I
)


FENCE_RE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*$")


def read_yaml(path: Path) -> dict:
    """Read a YAML file with plain-English errors. Ignores ```yaml lines accidentally pasted from a chat code box."""
    raw = path.read_text(encoding="utf-8-sig")
    text = "\n".join(l for l in raw.splitlines() if not FENCE_RE.match(l))
    rel = f"projects/{path.parent.name}/{path.name}"
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        lines = text.splitlines()
        where = ""
        if mark is not None:
            shown = lines[mark.line].strip() if mark.line < len(lines) else ""
            where = f" (line {mark.line + 1}: {shown[:100]!r})"
        problem = getattr(e, "problem", None) or "the layout is not valid YAML"
        raise SystemExit(
            f"{rel} has a formatting error{where}: {problem}.\n"
            "Common causes: a quote mark inside quoted text (write 'inches' instead of \"), a colon inside unquoted text, "
            "mixing [a, b] lists with '- item' lines under the same key, tabs instead of spaces, or ```yaml lines pasted from chat.\n"
            "Fix that line on GitHub (pencil icon) and run again.")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"{rel} must contain 'name: value' lines.")
    return data


@dataclass
class Project:
    asin: str
    dir: Path
    facts: dict = field(default_factory=dict)
    compare: dict | None = None   # set for comparison projects: {"products": [Project], "labels": [str], "config": {...}, "pick_ok": bool}

    @property
    def assets_dir(self) -> Path:
        return self.dir / "assets"

    @property
    def music_dir(self) -> Path:
        return self.dir / "music"

    @property
    def out_dir(self) -> Path:
        d = self.dir / "out"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def cache_dir(self) -> Path:
        d = self.dir / ".cache"
        d.mkdir(parents=True, exist_ok=True)
        return d


def _label(sub: "Project") -> str:
    f = sub.facts
    return str(f.get("short_name") or f.get("brand") or " ".join(str(f.get("product_name", sub.asin)).split()[:2])).strip()


def _load_compare(asin: str, pdir: Path, cfile: Path) -> "Project":
    conf = read_yaml(cfile)
    ids = [str(x).strip().upper() for x in (conf.get("products") or [])]
    if not 2 <= len(ids) <= 3:
        raise SystemExit(f"projects/{asin}/compare.yaml: 'products' must list 2 or 3 product folders (found {len(ids)}).")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"projects/{asin}/compare.yaml: the same product is listed twice.")
    subs = []
    for i in ids:
        if i == asin:
            raise SystemExit("compare.yaml cannot list its own folder.")
        sub = load_project(i)
        if sub.compare:
            raise SystemExit(f"{i} is itself a comparison project; list ordinary product folders.")
        subs.append(sub)
    labels = [_label(x) for x in subs]
    if len(set(l.lower() for l in labels)) != len(labels):
        raise SystemExit("Two products have the same short label. Give each facts.yaml a different short_name: (e.g. short_name: YESLIFE).")
    no_cut = sorted({str(a) for x in subs for a in (x.facts.get("assets_no_cutout") or [])})
    facts = {
        "asin": asin,
        "product_name": conf.get("title") or " vs ".join(labels),
        "category": conf.get("category") or subs[0].facts.get("category", ""),
        "audience": conf.get("audience") or subs[0].facts.get("audience", ""),
        "affiliate_url": subs[0].facts.get("affiliate_url", ""),
        "compare_points": conf.get("compare_points") or [],
        "notes": conf.get("notes", ""),
        "angle": "comparison",
        "assets_no_cutout": no_cut,
        "cinematic": not any(x.facts.get("cinematic", True) is False for x in subs),
    }
    pick_ok = all(len(x.facts.get("hands_on_notes") or []) >= 3 for x in subs)
    return Project(asin=asin, dir=pdir, facts=facts,
                   compare={"products": subs, "labels": labels, "config": conf, "pick_ok": pick_ok})


def load_project(asin: str, require_facts: bool = True) -> Project:
    asin = asin.strip().upper()
    if not ASIN_RE.match(asin):
        raise SystemExit(f"'{asin}' is not a valid project name (3-40 letters, digits, - or _).")
    pdir = ROOT / "projects" / asin
    if not pdir.exists():
        raise SystemExit(f"Folder projects/{asin}/ not found. Copy projects/_template/ to projects/{asin}/ first.")
    cfile = pdir / "compare.yaml"
    if cfile.exists():
        return _load_compare(asin, pdir, cfile)
    facts = {}
    fpath = pdir / "facts.yaml"
    if fpath.exists():
        facts = read_yaml(fpath)
    elif require_facts:
        raise SystemExit(f"projects/{asin}/facts.yaml is missing.")
    return Project(asin=asin, dir=pdir, facts=facts)


FACT_LISTS = ["manufacturer_claims", "specs", "hands_on_notes", "pros", "cons", "not_for"]


def fact_count(facts: dict) -> int:
    return sum(len(facts.get(k) or []) for k in FACT_LISTS)


def validate_facts(project: Project) -> tuple[list[str], list[str]]:
    """Return (errors, warnings)."""
    if project.compare:
        return _validate_compare(project)
    f = project.facts
    errors, warnings = [], []
    for key in ("product_name", "category", "audience", "affiliate_url"):
        if not f.get(key):
            errors.append(f"facts.yaml: '{key}' is required.")
    if str(f.get("asin", "")).upper() != project.asin:
        errors.append(f"facts.yaml: asin must equal the folder name ({project.asin}).")
    url = str(f.get("affiliate_url", ""))
    if url and not AMAZON_LINK_RE.match(url):
        errors.append(
            "facts.yaml: affiliate_url must be an Amazon link (amazon.*, amzn.to, amzn.eu or a.co). "
            "Third-party link shorteners/cloakers are not allowed in Associates."
        )
    if url and "amazon." in url and "tag=" not in url:
        warnings.append("affiliate_url has no tag=YOURID-20 parameter, so it will not earn commission.")
    n = fact_count(f)
    if n < 8:
        warnings.append(
            f"Only {n} facts supplied. A 5-9 minute script needs roughly 12+ real facts; "
            "with fewer, the writer has to lean on generic filler. Add specs, brand claims and your own hands-on notes."
        )
    if not f.get("hands_on_notes"):
        warnings.append(
            "No hands_on_notes. Without your own observations the video is a listing read-aloud, "
            "which is exactly what YouTube's 'inauthentic content' policy targets."
        )
    return errors, warnings


def _validate_compare(project: Project) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    subs = project.compare["products"]
    for sub in subs:
        e, w = validate_facts(sub)
        errors += [f"[{sub.asin}] {x}" for x in e]
        warnings += [f"[{sub.asin}] {x}" for x in w]
        n = len(sub.facts.get("hands_on_notes") or [])
        if 0 < n < 3:
            warnings.append(f"[{sub.asin}] only {n} hands_on_notes: the comparison will stay neutral and will not pick a winner.")
        if not any(p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".webm") for p in sub.assets_dir.glob("*")):
            errors.append(f"[{sub.asin}] has no photos/clips in its assets/ folder.")
    urls = [str(x.facts.get("affiliate_url", "")).strip() for x in subs]
    if len(set(urls)) != len(urls):
        errors.append("Two products share the same affiliate_url. Each product needs its own link.")
    if not project.compare["pick_ok"]:
        warnings.append("Not every product has 3+ hands_on_notes, so the verdict will say who each product suits instead of naming a winner.")
    if not project.facts.get("category"):
        errors.append("compare.yaml: add 'category:' (what kind of product these are).")
    return errors, warnings


# ---------------------------------------------------------------------------
# Hard rules on narration text (policy = must fix, style = warning)
# ---------------------------------------------------------------------------
POLICY_RULES = [
    (r"\$\s?\d|\b\d[\d,.]*\s?(usd|dollars?|bucks|euros?|pounds?)\b", "mentions a price"),
    (r"\b\d(\.\d)?\s*(out of 5|/5|stars?)\b|\bfive[- ]star\b|\bstar rating", "mentions a star rating"),
    (r"\b\d[\d,]*\s+(customer\s+)?(reviews|ratings)\b", "mentions a review/rating count"),
    (r"\b(customers?|reviewers?|buyers?|shoppers?)\s+(say|said|love|loved|report|reported|complain|praise|rave)", "paraphrases customer reviews"),
    (r"\bamazon'?s\s+choice\b|\bbest[- ]?sell(er|ing)\b|\b#\s?1\b", "uses an Amazon badge/ranking"),
    (r"\b(on sale|discounts?|coupons?|limited[- ]time|\d+\s?% off|percent off|price drop|lowest price|cheapest|free shipping|prime day|black friday|cyber monday)\b|\b(great|good|best|amazing|hot)\s+deals?\b", "uses price/promotion language"),
    (r"\b(walmart|target|best buy|ebay|aliexpress|temu|etsy)\b", "names another retailer"),
    (r"\b(cures?|heals?|clinically proven|medically proven|fda[- ]approved)\b", "makes a health claim"),
]
STYLE_RULES = [
    (r"\b(game[- ]changer|revolutionary|must[- ]have|life[- ]changing|miracle|guaranteed?)\b", "hype phrase"),
    (r"\bin today'?s video\b|\bsmash (that|the) like\b", "generic YouTube cliche"),
    (r"\bperfect for everyone\b|\bevery(one|body) needs\b", "overclaim"),
    (r"\b(better than|worse than|beats|outperforms?|clear winner|the winner|hands down)\b", "comparative claim: keep only if your own hands_on_notes support it"),
]


def check_text(text: str) -> tuple[list[str], list[str]]:
    policy, style = [], []
    for pat, why in POLICY_RULES:
        if re.search(pat, text, re.I):
            policy.append(why)
    for pat, why in STYLE_RULES:
        if re.search(pat, text, re.I):
            style.append(why)
    return policy, style
