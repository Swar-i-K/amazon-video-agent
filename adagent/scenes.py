"""Plan short AI-generated b-roll scenes (faceless, mood/lifestyle only) that a free-GPU notebook turns into clips.

The planner writes projects/<ASIN>/scenes.md. Rules baked in:
  * no visible faces (hands, silhouettes, people from behind, objects, places, light)
  * no product-performance visuals (no frost / ice / steam / instant results / before-after)
  * no text, logos, brand names, prices
  * at most 2 'product' scenes (animate a real product photo with a gentle camera move); the rest are 'mood'
"""
from __future__ import annotations

import re

from .common import extract_json, log
from .llm import ask
from .project import Project
from .scriptgen import assets_block, facts_yaml

MAX_PRODUCT_SCENES = 2

SCENE_SYSTEM = """You write shot descriptions for AI-generated b-roll in a faceless product video. A human reviews every scene, a video model renders each one as a 5-second clip, and the clips are cut together with real product photos.

HARD RULES
1. Every scene is FACELESS: show hands, silhouettes, people seen from behind or far away, objects, places and light. Never a visible face, portrait, or close-up of a person's face.
2. Mood and lifestyle only. Never show the product performing or any result: no frost, ice, steam, sparkles, glowing effects, instant effects, before/after, or comfort/health outcomes.
3. No text, logos, brand names, packaging, prices, screens or labels in the shot.
4. One simple action and one slow camera move per scene (push-in, drift, tilt, or a static frame with moving light).
5. 35 to 60 words per prompt: subject, setting, light, camera, mood. Cinematic commercial look.
6. type = "mood" for scenes without the product. type = "product" ONLY for at most 2 scenes that start from one of the AVAILABLE PHOTOS (gentle camera move and light only; the product must not change shape or do anything). For product scenes set ref to that photo's id (e.g. "A03"); for mood scenes ref is "none".
7. Scenes must match the product's real use (see FACTS) and must not imply features the FACTS do not state.
Output ONLY JSON."""

SCHEMA = '{"scenes": [{"id": "s01", "type": "mood|product", "ref": "none|A01", "prompt": "..."}]}'

FACE_WORDS = r"\b(face|faces|facial|smile|smiling|smiles|portrait|selfie|eyes?|lips|beaming|grinning|looking at (the )?camera)\b"
CLAIM_WORDS = (r"\b(frost|frosty|icy|ice|ice[- ]cold|steam|steaming|sparkl\w*|magic\w*|miracle|instant\w*|before and after|"
               r"cools? (down )?(instantly|immediately)|relief|sweat\w*|heat ?stroke|sunburn\w*)\b")
OTHER_BAD = r"(\$\s?\d|\b(price|sale|discount|deal|amazon|logo|brand|text overlay|caption|subtitle|label|packaging)\b)"


def check_prompt(text: str) -> list[str]:
    bad = []
    if re.search(FACE_WORDS, text, re.I):
        bad.append("mentions a face")
    if re.search(CLAIM_WORDS, text, re.I):
        bad.append("shows or implies a product effect")
    if re.search(OTHER_BAD, text, re.I):
        bad.append("mentions price/brand/text")
    return bad


def generate_scenes(project: Project, cfg: dict, entries: list[dict], n: int = 10) -> list[dict]:
    photos = [e for e in entries if e["usable"] and e["kind"] == "image"]
    user = (f"FACTS:\n{facts_yaml(project)}\n\nAVAILABLE PHOTOS:\n{assets_block(photos)}\n\n"
            f"Write {n} scenes (at most {MAX_PRODUCT_SCENES} of type product). Return JSON in this shape:\n{SCHEMA}")
    data = None
    for attempt in range(3):
        try:
            data = extract_json(ask(cfg["llm"]["model"], SCENE_SYSTEM, user, max_tokens=5000, temperature=0.8))
            if isinstance(data, dict) and isinstance(data.get("scenes"), list) and data["scenes"]:
                break
        except Exception as e:
            log(f"  scene reply unreadable ({e}); retrying")
        data = None
        user += "\n\nIMPORTANT: reply with ONLY one compact JSON object in the shape above."
    if not data:
        raise RuntimeError("The model did not return usable scenes after 3 tries. Try again or choose another model.")
    by_id = {e["id"]: e for e in photos}
    scenes, n_prod = [], 0
    for i, s in enumerate(data["scenes"], 1):
        prompt = re.sub(r"\s+", " ", str(s.get("prompt", ""))).strip()
        if not prompt:
            continue
        issues = check_prompt(prompt)
        if issues:
            log(f"  dropped scene {i}: {', '.join(issues)}")
            continue
        typ, ref = str(s.get("type", "mood")).lower(), str(s.get("ref", "none"))
        if typ == "product":
            if ref not in by_id or n_prod >= MAX_PRODUCT_SCENES:
                typ, ref = "mood", "none"
                log(f"  scene {i}: product scene converted to mood (bad reference or too many)")
            else:
                n_prod += 1
        else:
            typ, ref = "mood", "none"
        scenes.append({"id": f"s{len(scenes) + 1:02d}", "type": typ, "ref": by_id[ref]["file"] if typ == "product" else "none",
                       "prompt": prompt})
    return scenes


def scenes_to_md(scenes: list[dict], project: Project) -> str:
    L = [f"# SCENES for {project.asin}  (AI-generated clips, about 5 seconds each)",
         "# Next: run notebooks/make_ai_clips.ipynb on a free GPU (Kaggle or Colab). It writes ai_s01.mp4 ... into assets/.",
         "# Edit prompts freely. Keep every scene faceless and free of product-performance claims. Delete a line to skip a scene.",
         "# Format:  [id] type: mood|product | ref: none|photo file name | prompt: ...",
         ""]
    for s in scenes:
        L.append(f"[{s['id']}] type: {s['type']} | ref: {s['ref']} | prompt: {s['prompt']}")
    return "\n".join(L) + "\n"


def md_to_scenes(md: str) -> list[dict]:
    out = []
    for line in md.splitlines():
        m = re.match(r"^\[(\w+)\]\s*type:\s*(\w+)\s*\|\s*ref:\s*([^|]+?)\s*\|\s*prompt:\s*(.+)$", line.strip())
        if m:
            out.append({"id": m.group(1), "type": m.group(2).lower(), "ref": m.group(3).strip(), "prompt": m.group(4).strip()})
    return out
