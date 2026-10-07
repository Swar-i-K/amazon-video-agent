"""Command line:  python -m adagent <command> ASIN [options]"""
from __future__ import annotations

import argparse
import re
import shutil
import sys

from . import assets as A
from . import finish as F
from . import render as R
from . import scriptgen as S
from . import timeline as T
from . import tts as V
from .common import load_config, log
from .overlay import build_overlays
from .project import check_text, load_project, validate_facts


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48] or "video"


def _prep(args, need_vision=True):
    cfg = load_config()
    for k in ("width", "height", "fps", "crf"):
        if getattr(args, k, None):
            cfg["video"][k] = getattr(args, k)
    project = load_project(args.asin)
    errors, warns = validate_facts(project)
    for w in warns:
        log(f"  WARNING: {w}")
    if errors:
        raise SystemExit("\n".join(["Fix these in facts.yaml first:"] + [f"  - {e}" for e in errors]))
    log("Checking assets ...")
    entries = A.build_index(project, cfg, vision=not getattr(args, "no_vision", False))
    for e in entries:
        if not e["usable"]:
            log(f"  EXCLUDED {e['file']}: {e['reason']}")
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

    cov, cov_notes = A.coverage_report(entries, tl.total)
    for n in cov_notes:
        log(f"  {n}")

    out = project.out_dir
    voice_wav = out / "voice.wav"
    T.write_voice(tl, voice_wav)

    W, H = cfg["video"]["width"], cfg["video"]["height"]
    shots = R.plan_shots(tl.rows, tl.total, entries, project.assets_dir, cfg, seed=project.asin)
    extra = [vinfo["attribution"]] if vinfo.get("attribution") else []
    ovs = build_overlays(tl.rows, tl.total, cfg, W, H, project.facts["product_name"], extra)
    silent = out / "_video_only.mp4"
    log(f"Rendering {len(shots)} shots at {W}x{H} ...")
    R.render_video(shots, ovs, tl.total, cfg, silent, out / "ffmpeg.log", max_seconds=args.max_seconds)

    final = out / f"{project.asin}_{_slug(script.get('title') or project.facts['product_name'])}.mp4"
    music = F.find_music(project)
    F.mux(silent, voice_wav, final, args.max_seconds or tl.total, music)
    silent.unlink(missing_ok=True)
    F.write_description(project, cfg, script, tl, vinfo, out / "description.txt")
    F.write_review(project, script, entries, cov_notes, warns, tl, final, voice_key, out / "REVIEW_CHECKLIST.md")
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
        if script:
            sp.add_argument("--angle", choices=list(S.ANGLES), help="video angle (default: rotates automatically)")
            sp.add_argument("--force", action="store_true", help="overwrite script.md / ignore daily limit")
        if voice:
            sp.add_argument("--voice", help="voice key from config.yaml (see 'voices')")
        if render:
            sp.add_argument("--no-length-check", action="store_true")
            sp.add_argument("--max-seconds", type=float, help="render only the first N seconds (preview)")
            for k in ("width", "height", "fps", "crf"):
                sp.add_argument(f"--{k}", type=int)

    common(sub.add_parser("index", help="check and describe your assets"))
    common(sub.add_parser("script", help="write script.md for review"), script=True)
    common(sub.add_parser("render", help="voice + render from script.md"), voice=True, render=True)
    common(sub.add_parser("all", help="script + render in one go"), voice=True, script=True, render=True)
    sub.add_parser("voices", help="list configured voices")
    sub.add_parser("eleven-voices", help="list voices available to your ElevenLabs key")

    args = p.parse_args(argv)
    {"index": cmd_index, "script": cmd_script, "render": cmd_render, "all": cmd_all,
     "voices": cmd_voices, "eleven-voices": cmd_eleven_voices}[args.cmd](args)


if __name__ == "__main__":
    main(sys.argv[1:])
