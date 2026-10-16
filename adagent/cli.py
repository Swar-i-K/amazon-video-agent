"""Command line:  python -m adagent <command> ASIN [options]"""
from __future__ import annotations

import argparse
import re
import shutil
import sys

from . import assets as A
from . import finish as F
from . import render as R
from . import scenes as SC
from . import scriptgen as S
from . import timeline as T
from . import tts as V
from .common import load_config, log
from .overlay import build_overlays
from .templates import apply_template, tpl_of
from .project import check_text, load_project, validate_facts


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48] or "video"


def choose_length(cfg: dict, project, choice) -> str:
    """Length preset by video type (review / compare / list), or 'short' for any type. Every preset stays under 9 minutes."""
    presets = cfg.get("length_presets", {})
    if choice == "short":
        key = "short"
    else:
        key = "list" if (project.compare and project.compare["kind"] == "list") else "compare" if project.compare else "review"
    if choice not in (None, "auto", "standard", "short"):
        raise SystemExit(f"Unknown length '{choice}'. Options: auto, short")
    if presets.get(key):
        cfg["length"].update(presets[key])
    return key


def _prep(args, need_vision=True):
    cfg = load_config()
    for k in ("width", "height", "fps", "crf"):
        if getattr(args, k, None):
            cfg["video"][k] = getattr(args, k)
    project = load_project(args.asin)
    key = choose_length(cfg, project, getattr(args, "length", None))
    log(f"Length preset: {key} ({cfg['length']['min_minutes']}-{cfg['length']['max_minutes']} min, aim {cfg['length']['target_minutes']})")
    tpl = apply_template(cfg, getattr(args, "template", None))
    if getattr(args, "template", None) or args.__dict__.get("cmd") in ("render", "all", "preview"):
        log(f"Template: {tpl['key']} ({tpl.get('label', '')})")
    errors, warns = validate_facts(project)
    for w in warns:
        log(f"  WARNING: {w}")
    if errors:
        raise SystemExit("\n".join(["Fix these in facts.yaml first:"] + [f"  - {e}" for e in errors]))
    log("Checking assets ...")
    entries = A.build_index_for(project, cfg, vision=not getattr(args, "no_vision", False))
    for e in entries:
        if not e["usable"]:
            log(f"  EXCLUDED {e.get('label', '')} {e['file']}: {e['reason']}".replace("  EXCLUDED  ", "  EXCLUDED "))
    log(f"  {sum(e['usable'] for e in entries)} of {len(entries)} files usable")
    return cfg, project, entries, warns


def cmd_script(args):
    cfg, project, entries, _ = _prep(args)
    target = project.dir / "script.md"
    if target.exists() and not args.force:
        raise SystemExit(f"{target} already exists (it may contain your edits). Use --force to overwrite.")
    script = S.make_script(project, cfg, entries, angle=args.angle, force=args.force)
    md = S.script_to_md(script, project)
    target.write_text(md, encoding="utf-8")
    (project.out_dir / "script.md").write_text(md, encoding="utf-8")
    S.save_archive(project, script)
    rv = script["review"]
    log(f"\nScript written to projects/{project.asin}/script.md  (~{rv['estimated_minutes']} min)")
    log(f"  open fact-check issues: {len(rv['fact_issues'])}, policy flags: {len(rv['policy'])}, style flags: {len(rv['style'])}")
    if rv.get("length_warning"):
        log("\n  !! " + rv["length_warning"])
    log("  Review/edit the file, then run the render step.")


def cmd_scenes(args):
    cfg, project, entries, _ = _prep(args)
    if project.compare:
        raise SystemExit("AI scenes are made per product. Run 'scenes' on each product folder; the AI clips (ai_*.mp4) in a product's "
                         "assets/ are used automatically by comparison and list videos that include that product.")
    target = project.dir / "scenes.md"
    if target.exists() and not args.force:
        raise SystemExit(f"{target} already exists (it may contain your edits). Use --force to overwrite.")
    scenes = SC.generate_scenes(project, cfg, entries, n=args.count or int(cfg.get("scenes", {}).get("count", 10)))
    if len(scenes) < 3:
        raise SystemExit("Fewer than 3 safe scenes survived the checks. Run it again, or add more detail to facts.yaml (use_cases).")
    md = SC.scenes_to_md(scenes, project)
    target.write_text(md, encoding="utf-8")
    (project.out_dir / "scenes.md").write_text(md, encoding="utf-8")
    n_prod = sum(1 for s in scenes if s["type"] == "product")
    log(f"\n{len(scenes)} scene(s) written to projects/{project.asin}/scenes.md ({n_prod} product, {len(scenes) - n_prod} mood).")
    log("  Review/edit it, then run notebooks/make_ai_clips.ipynb on a free GPU (Kaggle or Colab).")
    log("  The clips arrive in assets/ as ai_s01.mp4 ... and are used automatically on the next render.")


def cmd_preview(args):
    """One sample picture per template, so you can compare the looks before rendering a whole video."""
    from .preview import render_template_preview
    args.no_vision = True   # the preview only needs a photo; the face/price screening still runs on real renders
    cfg, project, entries, _ = _prep(args)
    out = render_template_preview(project, cfg, entries, only=getattr(args, "template", None))
    log(f"\nTemplate preview saved: {out}")


def cmd_render(args):
    cfg, project, entries, warns = _prep(args)
    src = project.dir / "script.md"
    if not src.exists():
        raise SystemExit("No script.md yet. Run the 'script' step first (or use 'all').")
    script = S.md_to_script(src.read_text(encoding="utf-8"))
    run_render(args, cfg, project, entries, warns, script)


def cmd_all(args):
    cfg, project, entries, warns = _prep(args)
    script = S.make_script(project, cfg, entries, angle=args.angle, force=args.force)
    md = S.script_to_md(script, project)
    (project.dir / "script.md").write_text(md, encoding="utf-8")
    (project.out_dir / "script.md").write_text(md, encoding="utf-8")
    S.save_archive(project, script)
    run_render(args, cfg, project, entries, warns, script)


def run_render(args, cfg, project, entries, warns, script):
    if not script["segments"]:
        raise SystemExit("The script has no lines.")
    segs = S.assemble(script, project, cfg)

    bad = []
    for s in segs:
        pol, _ = check_text(s["text"] + " " + s.get("onscreen", ""))
        if pol:
            bad.append(f"  [{s['id']}] {', '.join(pol)}: {s['text'][:80]}")
    if bad:
        raise SystemExit("These lines break Amazon Associates / content rules. Edit script.md and retry:\n" + "\n".join(bad))

    if args.max_seconds:  # preview: only voice the opening, so no credits are spent on the rest
        keep, acc = [], cfg["length"]["lead_in"]
        for s in segs:
            keep.append(s)
            acc += len(s["text"].split()) / (cfg["length"]["wpm"] / 60.0) + cfg["voice"]["pause_sentence"]
            if acc >= args.max_seconds + 6:
                break
        segs, args.no_length_check = keep, True
        log(f"PREVIEW: first {args.max_seconds:.0f}s only ({len(segs)} lines voiced).")
    usable = [e for e in entries if e["usable"]]
    if not usable:  # stop BEFORE any voice credits are spent
        why = [f"  - {e['file']}: {e['reason']}" for e in entries if not e["usable"]]
        found = len(entries)
        raise SystemExit(
            f"No usable footage in projects/{project.asin}/assets/ ({found} file(s) found).\n"
            + ("\n".join(["These were excluded:"] + why) + "\n" if why else
               "The folder has no photos or clips. Photos must be .jpg .jpeg .png .webp and clips .mp4 .mov .webm, "
               "placed directly in that folder (not in a sub-folder), and the folder name must match the ASIN exactly.\n")
            + "Nothing was spent on voice. Fix the footage and run render again.")
    est = S.estimate_minutes(segs, cfg)
    lo, hi = cfg["length"]["min_minutes"], cfg["length"]["max_minutes"]
    log(f"Estimated length {est:.1f} min (allowed {lo}-{hi}).")
    if not args.no_length_check and not (lo + 0.1 <= est <= hi - 0.1):
        if est < lo:
            more = int((lo + 0.2 - est) * 60 * cfg["length"]["wpm"] / 60)
            raise SystemExit(f"Script is only ~{est:.1f} min (minimum {lo:.0f}). Add about {more} more words "
                             f"(~{more // 22} lines) to projects/{project.asin}/script.md, or add facts to facts.yaml "
                             "and re-run the script step with 'force' ticked.")
        raise SystemExit(f"Script is ~{est:.1f} min (maximum {hi:.0f}). Delete some lines from script.md.")

    voice_key, _ = V.resolve_voice(cfg, args.voice)
    log(f"Voice: {voice_key}")
    vinfo = V.preflight(cfg, voice_key, segs)
    warns = list(warns) + vinfo["warnings"]
    for w in vinfo["warnings"]:
        log(f"  NOTE: {w}")
    arrays = V.synthesize_all(project, cfg, voice_key, segs)
    tl = T.build_timeline(segs, arrays, cfg, enforce=not args.no_length_check)
    log(f"Timeline: {tl.total / 60:.2f} min")

    if cfg["disclosure"].get("mode", "full") == "description_only":
        warns.append("Disclosure mode is description_only: there is no disclosure inside the video. It is in description.txt (first line) and "
                     "pinned_comment.txt. The FTC says disclosure inside the video is clearer; on_screen mode adds silent text on screen.")
        log("  NOTE: disclosure mode = description_only (nothing in the video; use description.txt and pinned_comment.txt)")
    cov, cov_notes = A.coverage_report(entries, tl.total)
    for n in cov_notes:
        log(f"  {n}")

    out = project.out_dir
    voice_wav = out / "voice.wav"
    T.write_voice(tl, voice_wav)

    W, H = cfg["video"]["width"], cfg["video"]["height"]
    shots = R.plan_shots(tl.rows, tl.total, entries, project.assets_dir, cfg, seed=project.asin, facts=project.facts)
    cine_files = sorted({s.asset["file"] for s in shots if getattr(s, "cine", None) is not None})
    if cine_files:
        n_c = sum(1 for s in shots if s.cine is not None)
        cov_notes.append(f"Cinematic product shots: {n_c} shot(s) from {', '.join(cine_files)}. Check the cut-out edges "
                         "in the video; if one looks rough, add its file name to assets_no_cutout: in facts.yaml.")
        log(f"  cinematic shots: {n_c} (from {len(cine_files)} photo(s))")
    extra = [vinfo["attribution"]] if vinfo.get("attribution") else []
    ai_spans = [(s.start, s.start + s.dur) for s in shots if s.asset["file"].lower().startswith("ai_")]
    if ai_spans:
        vinfo["ai_scenes"] = True
        warns.append("This video contains AI-generated footage. In YouTube's upload form, answer YES to the altered/synthetic "
                     "content question (realistic AI scenes must be labelled). Review each AI clip for odd hands, distorted objects "
                     "or anything that looks like a product claim.")
        log(f"  AI-generated clips in {len(ai_spans)} shot(s); on-screen tag and description note added")
    kind = project.compare["kind"] if project.compare else None
    rank_notes = ([rl or bf for rl, bf in zip(project.compare["rank_labels"], project.compare["best_for"])] if kind == "list" else None)
    ovs = build_overlays(tl.rows, tl.total, cfg, W, H, project.facts["product_name"], extra, ai_spans=ai_spans,
                         product_labels=project.compare["labels"] if project.compare else None, kind=kind, rank_notes=rank_notes,
                         endcard_line={"list": "Links to every product in the description",
                                       "compare": "Links to each product in the description"}.get(kind, "Link in the description"))
    silent = out / "_video_only.mp4"
    log(f"Rendering {len(shots)} shots at {W}x{H} ...")
    R.render_video(shots, ovs, tl.total, cfg, silent, out / "ffmpeg.log", max_seconds=args.max_seconds)

    final = out / f"{project.asin}_{_slug(script.get('title') or project.facts['product_name'])}.mp4"
    music = F.find_music(project)
    F.mux(silent, voice_wav, final, args.max_seconds or tl.total, music)
    silent.unlink(missing_ok=True)
    F.write_description(project, cfg, script, tl, vinfo, out / "description.txt")
    F.write_review(project, script, entries, cov_notes, warns, tl, final, voice_key, out / "REVIEW_CHECKLIST.md",
                   disclosure_mode=cfg["disclosure"].get("mode", "full"))
    if (project.dir / "script.md").exists():
        shutil.copy(project.dir / "script.md", out / "script_used.md")
    if not args.max_seconds:
        S.save_archive(project, script)
    log(f"\nDone: {final}\nAlso: description.txt, REVIEW_CHECKLIST.md")


def cmd_index(args):
    _prep(args)


def cmd_voices(args):
    cfg = load_config()
    for k, v in cfg["voice"]["options"].items():
        print(f"{k:16} {v['label']}")


def cmd_eleven_voices(args):
    for v in V.eleven_list_voices():
        g = (v.get("labels") or {}).get("gender", "")
        print(f"{v['voice_id']}  {v['name']:<22} {v['category'] or '':<12} {g}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="adagent", description="Faceless Amazon product-ad video builder (you supply the footage).")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, voice=False, script=False, render=False):
        sp.add_argument("asin")
        sp.add_argument("--no-vision", action="store_true", help="skip AI image checks (tests only)")
        if script or render:
            sp.add_argument("--length", choices=["auto", "standard", "short"], help="auto = by video type, always under 9 min (default); short = 1.5-4.8 min")
        if script:
            sp.add_argument("--angle", choices=list(S.ANGLES), help="video angle (default: rotates automatically)")
            sp.add_argument("--force", action="store_true", help="overwrite script.md / ignore daily limit")
        if voice:
            sp.add_argument("--voice", help="voice key from config.yaml (see 'voices')")
        if render:
            sp.add_argument("--template", help="visual template: one, two, three, four or best (default from config.yaml)")
            sp.add_argument("--no-length-check", action="store_true")
            sp.add_argument("--max-seconds", type=float, help="render only the first N seconds (preview)")
            for k in ("width", "height", "fps", "crf"):
                sp.add_argument(f"--{k}", type=int)

    sp = sub.add_parser("scenes", help="plan AI b-roll scenes (scenes.md) for the free-GPU notebook")
    common(sp, script=False)
    sp.add_argument("--force", action="store_true", help="overwrite scenes.md")
    sp.add_argument("--count", type=int, default=None, help="how many scenes to plan (default from config.yaml)")
    sp2 = sub.add_parser("preview", help="one sample picture per visual template")
    common(sp2)
    sp2.add_argument("--template", help="only this template (default: all five)")
    common(sub.add_parser("index", help="check and describe your assets"))
    common(sub.add_parser("script", help="write script.md for review"), script=True)
    common(sub.add_parser("render", help="voice + render from script.md"), voice=True, render=True)
    common(sub.add_parser("all", help="script + render in one go"), voice=True, script=True, render=True)
    sub.add_parser("voices", help="list configured voices")
    sub.add_parser("eleven-voices", help="list voices available to your ElevenLabs key")

    args = p.parse_args(argv)
    {"index": cmd_index, "script": cmd_script, "render": cmd_render, "all": cmd_all, "scenes": cmd_scenes, "preview": cmd_preview,
     "voices": cmd_voices, "eleven-voices": cmd_eleven_voices}[args.cmd](args)


if __name__ == "__main__":
    main(sys.argv[1:])
