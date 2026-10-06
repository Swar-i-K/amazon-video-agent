"""Project folder handling, facts validation, and hard content rules."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .common import ROOT

ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")
AMAZON_LINK_RE = re.compile(
    r"^https://((www\.)?amazon\.[a-z.]{2,6}/|amzn\.(to|eu|asia)/|a\.co/)", re.I
)


@dataclass
class Project:
    asin: str
    dir: Path
    facts: dict = field(default_factory=dict)

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


def load_project(asin: str, require_facts: bool = True) -> Project:
    asin = asin.strip().upper()
    if not ASIN_RE.match(asin):
        raise SystemExit(f"'{asin}' is not a valid ASIN (10 letters/digits).")
    pdir = ROOT / "projects" / asin
    if not pdir.exists():
        raise SystemExit(f"Folder projects/{asin}/ not found. Copy projects/_template/ to projects/{asin}/ first.")
    facts = {}
    fpath = pdir / "facts.yaml"
    if fpath.exists():
        facts = yaml.safe_load(fpath.read_text(encoding="utf-8")) or {}
    elif require_facts:
        raise SystemExit(f"projects/{asin}/facts.yaml is missing.")
    return Project(asin=asin, dir=pdir, facts=facts)


FACT_LISTS = ["manufacturer_claims", "specs", "hands_on_notes", "pros", "cons", "not_for"]


def fact_count(facts: dict) -> int:
    return sum(len(facts.get(k) or []) for k in FACT_LISTS)


def validate_facts(project: Project) -> tuple[list[str], list[str]]:
    """Return (errors, warnings)."""
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
