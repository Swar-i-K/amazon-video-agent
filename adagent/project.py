"""Project folder handling, facts validation, and hard content rules."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .common import ROOT, load_yaml_strict

ASIN_RE = re.compile(r"^[A-Z0-9][A-Z0-9_-]{2,39}$")   # a product ASIN, or the name of a comparison folder
AMAZON_LINK_RE = re.compile(
    r"^https://((www\.)?amazon\.[a-z.]{2,6}/|amzn\.(to|eu|asia)/|a\.co/)", re.I
)


def normalize_keys(d: dict) -> dict:
    """Accept 'Product Name:' / 'PRODUCT_NAME:' for product_name, and unwrap one wrapper heading such as 'facts:' or 'product:'."""
    def norm(k):
        return re.sub(r"[\s-]+", "_", str(k).strip().lower())
    out = {norm(k): v for k, v in d.items()}
    if len(out) == 1:
        (only,) = out.values()
        if isinstance(only, dict) and any(k in {norm(x) for x in only} for k in ("product_name", "affiliate_url", "asin")):
            out = {norm(k): v for k, v in only.items()}
    return out


FENCE_RE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*$")


def read_yaml(path: Path) -> dict:
    """Read a YAML file with plain-English errors. Ignores ```yaml lines accidentally pasted from a chat code box."""
    raw = path.read_text(encoding="utf-8-sig")
    text = "\n".join(l for l in raw.splitlines() if not FENCE_RE.match(l))
    rel = f"projects/{path.parent.name}/{path.name}"
    try:
        data = load_yaml_strict(text)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        lines = text.splitlines()
        where = ""
        if mark is not None:
            shown = lines[mark.line].strip() if mark.line < len(lines) else ""
            where = f" (line {mark.line + 1}: {shown[:100]!r})"
        problem = getattr(e, "problem", None) or "the layout is not valid YAML"
        if "appears twice" in problem:
            raise SystemExit(f"{rel} has a formatting error{where}: {problem}.\n"
                             "Merge the two into one (for example, put all the list items under a single 'hands_on_notes:'), then run again.")
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


def _keywords(v) -> list[str]:
    """Accepts a list or a comma-separated string; strips, de-duplicates, keeps order."""
    if not v:
        return []
    items = re.split(r"[,\n]", v) if isinstance(v, str) else list(v)
    out, seen = [], set()
    for k in items:
        k = str(k).strip()
        if k and k.lower() not in seen:
            seen.add(k.lower())
            out.append(k)
    return out


LIMITS = {"compare": (2, 3), "list": (3, 9)}


def _load_compare(asin: str, pdir: Path, cfile: Path) -> "Project":
    kind = "list" if cfile.name == "list.yaml" else "compare"
    other = pdir / ("compare.yaml" if kind == "list" else "list.yaml")
    if other.exists():
        raise SystemExit(f"projects/{asin} has both compare.yaml and list.yaml. Keep only one.")
    conf = normalize_keys(read_yaml(cfile))
    raw = conf.get("products") or []
    entries = []
    for item in raw:
        if isinstance(item, dict):
            entries.append({"id": str(item.get("id") or item.get("asin") or "").strip().upper(),
                            "label": str(item.get("label") or "").strip(), "best_for": str(item.get("best_for") or "").strip()})
        else:
            entries.append({"id": str(item).strip().upper(), "label": "", "best_for": ""})
    ids = [e["id"] for e in entries]
    lo, hi = LIMITS[kind]
    if not lo <= len(ids) <= hi:
        raise SystemExit(f"projects/{asin}/{cfile.name}: 'products' must list {lo} to {hi} product folders (found {len(ids)}).")
    if len(set(ids)) != len(ids) or "" in ids:
        raise SystemExit(f"projects/{asin}/{cfile.name}: a product is listed twice or is empty.")
    subs = []
    for i in ids:
        if i == asin:
            raise SystemExit(f"{cfile.name} cannot list its own folder.")
        sub = load_project(i)
        if sub.compare:
            raise SystemExit(f"{i} is itself a comparison/list project; list ordinary product folders.")
        subs.append(sub)
    labels = [_label(x) for x in subs]
    if len(set(l.lower() for l in labels)) != len(labels):
        raise SystemExit("Two products have the same short label. Give each facts.yaml a different short_name: (e.g. short_name: YESLIFE).")
    no_cut = sorted({str(a) for x in subs for a in (x.facts.get("assets_no_cutout") or [])})
    order = str(conf.get("order", "countdown")).strip().lower()
    if order not in ("countdown", "ascending"):
        raise SystemExit(f"projects/{asin}/{cfile.name}: order must be 'countdown' (N first, #1 last) or 'ascending'.")
    kw = _keywords(conf.get("keywords")) or _keywords([k for x in subs for k in _keywords(x.facts.get("keywords"))])[:12]
    facts = {
        "asin": asin,
        "product_name": conf.get("title") or (" vs ".join(labels) if kind == "compare" else f"Top {len(subs)} {subs[0].facts.get('category', '')}".strip()),
        "category": conf.get("category") or subs[0].facts.get("category", ""),
        "audience": conf.get("audience") or subs[0].facts.get("audience", ""),
        "affiliate_url": subs[0].facts.get("affiliate_url", ""),
        "compare_points": conf.get("compare_points") or [],
        "notes": conf.get("notes", ""),
        "keywords": kw,
        "angle": kind,
        "assets_no_cutout": no_cut,
        "cinematic": not any(x.facts.get("cinematic", True) is False for x in subs),
    }
    pick_ok = all(len(x.facts.get("hands_on_notes") or []) >= 3 for x in subs)
    return Project(asin=asin, dir=pdir, facts=facts, compare={
        "kind": kind, "products": subs, "labels": labels, "config": conf, "pick_ok": pick_ok,
        "ranks": list(range(1, len(subs) + 1)), "rank_labels": [e["label"] for e in entries],
        "best_for": [e["best_for"] for e in entries], "order": order, "criteria": str(conf.get("criteria", "") or "").strip()})


def load_project(asin: str, require_facts: bool = True) -> Project:
    asin = asin.strip().upper()
    if not ASIN_RE.match(asin):
        raise SystemExit(f"'{asin}' is not a valid project name (3-40 letters, digits, - or _).")
    pdir = ROOT / "projects" / asin
    if not pdir.exists():
        raise SystemExit(f"Folder projects/{asin}/ not found. Copy projects/_template/ to projects/{asin}/ first.")
    for name in ("compare.yaml", "list.yaml"):
        cfile = pdir / name
        if cfile.exists():
            return _load_compare(asin, pdir, cfile)
    facts = {}
    fpath = pdir / "facts.yaml"
    if fpath.exists():
        facts = normalize_keys(read_yaml(fpath))
        facts["keywords"] = _keywords(facts.get("keywords"))
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
    missing = [k for k in ("product_name", "category", "audience", "affiliate_url") if not f.get(k)]
    for key in missing:
        errors.append(f"facts.yaml: '{key}' is required.")
    if len(missing) >= 3:
        found = ", ".join(sorted(map(str, f))) or "(nothing)"
        errors.append(f"The lines I found at the top of projects/{project.asin}/facts.yaml are: {found}. "
                      "Each required line must start at the very left edge, like  product_name: \"...\"  (no spaces before it, "
                      "no # at the start). Copy projects/_template/facts.yaml and fill it in.")
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
            f"Only {n} facts supplied. A full script needs roughly 12+ real facts; "
            "with fewer, the writer has to lean on generic filler. Add specs, brand claims and your own hands-on notes."
        )
    if len(f.get("keywords") or []) > 30:
        warnings.append("More than 30 keywords: keep the best 10-15, long lists read as keyword stuffing.")
    if not f.get("hands_on_notes"):
        warnings.append(
            "No hands_on_notes. Without your own observations the video is a listing read-aloud, "
            "which is exactly what YouTube's 'inauthentic content' policy targets."
        )
    return errors, warnings


def _validate_compare(project: Project) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    c = project.compare
    subs = c["products"]
    for sub in subs:
        e, w = validate_facts(sub)
        errors += [f"[{sub.asin}] {x}" for x in e]
        warnings += [f"[{sub.asin}] {x}" for x in w]
        n = len(sub.facts.get("hands_on_notes") or [])
        if 0 < n < 3:
            warnings.append(f"[{sub.asin}] only {n} hands_on_notes: " + (
                "its list entry will rest on facts only, not on your own tests." if c["kind"] == "list"
                else "the comparison will stay neutral and will not pick a winner."))
        if not any(p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".webm") for p in sub.assets_dir.glob("*")):
            errors.append(f"[{sub.asin}] has no photos/clips in its assets/ folder.")
    urls = [str(x.facts.get("affiliate_url", "")).strip() for x in subs]
    if len(set(urls)) != len(urls):
        errors.append("Two products share the same affiliate_url. Each product needs its own link.")
    if c["kind"] == "compare" and not c["pick_ok"]:
        warnings.append("Not every product has 3+ hands_on_notes, so the verdict will say who each product suits instead of naming a winner.")
    if c["kind"] == "list":
        if not c["criteria"]:
            warnings.append("list.yaml has no 'criteria:' (how you chose and ranked these). Add 1-2 honest sentences; without them the script can only say it is your order.")
        warnings.append("The ranking order is YOUR decision: put your best pick first in 'products:'. The script will present it as your pick, never as a test result.")
    if not project.facts.get("category"):
        errors.append(f"{'list' if c['kind'] == 'list' else 'compare'}.yaml: add 'category:' (what kind of product these are).")
    if len(project.facts.get("keywords") or []) > 30:
        warnings.append("More than 30 keywords: keep the best 10-15, long lists read as keyword stuffing.")
    return errors, warnings


# ---------------------------------------------------------------------------
# Hard rules on narration text (policy = must fix, style = warning)
# ---------------------------------------------------------------------------
POLICY_RULES = [
    (r"\bhands[- ]on notes?\b|\bthe notes (say|mention|state)\b|\baccording to (my|the) (notes|facts)\b", "refers to your notes/facts out loud (speak in the first person or say 'the brand says')"),
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
    (r"\b(cheap|cheaper|cheapest|affordable|budget|pricey|expensive|bargain)\b", "price wording: prices change and are not allowed to be quoted, so avoid describing price level"),
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
