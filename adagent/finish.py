"""Final mix (voice + optional ducked music), description text, and the human review checklist."""
from __future__ import annotations

from pathlib import Path

from .common import fmt_ts, media_duration, run
from .project import Project
from .timeline import Timeline, chapters

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


def find_music(project: Project) -> Path | None:
    if project.music_dir.exists():
        for p in sorted(project.music_dir.glob("*")):
            if p.suffix.lower() in AUDIO_EXT:
                return p
    return None


def mux(video: Path, voice_wav: Path, out: Path, total: float, music: Path | None) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", voice_wav]
    if music:
        cmd += ["-stream_loop", "-1", "-i", music]
        fc = ("[1:a]loudnorm=I=-16:TP=-1.5:LRA=11,asplit=2[vsc][vmix];"
              f"[2:a]loudnorm=I=-30:TP=-3:LRA=7,afade=t=in:d=2,afade=t=out:st={max(0, total - 3):.2f}:d=3[m];"
              "[m][vsc]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400[duck];"
              "[vmix][duck]amix=inputs=2:duration=first:normalize=0[a]")
    else:
        fc = "[1:a]loudnorm=I=-16:TP=-1.5:LRA=11[a]"
    cmd += ["-filter_complex", fc, "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-t", f"{total:.3f}", "-movflags", "+faststart", out]
    run(cmd)


def write_description(project: Project, cfg: dict, script: dict, tl: Timeline, voice_info: dict, path: Path) -> None:
    """Writes title.txt, description.txt (disclosure on the first line) and pinned_comment.txt next to `path`."""
    f, d = project.facts, cfg["disclosure"]
    title = script.get("title") or f["product_name"]
    disc = d["on_screen"] + " This description and video contain affiliate links."
    L = [disc, "", script.get("description_intro", ""), ""]
    if project.compare and project.compare["kind"] == "list":
        L.append("Every product in this list on Amazon:")
        for i, (sub, lab) in enumerate(zip(project.compare["products"], project.compare["labels"]), 1):
            L.append(f"#{i} {lab} ({sub.facts['product_name']}): {sub.facts['affiliate_url']}")
        links_line = "Links to every product are in the description."
    elif project.compare:
        L.append("Products compared on Amazon:")
        for i, (sub, lab) in enumerate(zip(project.compare["products"], project.compare["labels"])):
            L.append(f"{'ABC'[i]}. {lab} ({sub.facts['product_name']}): {sub.facts['affiliate_url']}")
        links_line = "Links to each product are in the description."
    else:
        L.append(f"Product on Amazon: {f['affiliate_url']}")
        links_line = "The product link is in the description."
    L.append("")
    note = [d["ai_voice_note"]]
    if voice_info.get("ai_scenes"):
        note.append(d.get("ai_scenes_note", "Some scenes in this video are AI-generated."))
    if voice_info.get("attribution"):
        note.append(voice_info["attribution"] + ".")
    L += [" ".join(note), ""]
    ch = chapters(tl, project.compare["labels"] if project.compare else None)
    if ch:
        L += ["Chapters"] + [f"{fmt_ts(t)} {name}" for t, name in ch] + [""]
    if script.get("tags"):
        L += ["Tags: " + ", ".join(script["tags"])]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    (path.parent / "title.txt").write_text(title + "\n", encoding="utf-8")
    (path.parent / "pinned_comment.txt").write_text(f"{d['on_screen']} {links_line}\n", encoding="utf-8")


def write_review(project: Project, script: dict, entries: list[dict], cov_notes: list[str], warnings: list[str],
                 tl: Timeline, final_path: Path, voice_key: str, path: Path, disclosure_mode: str = "full") -> None:
    rv = script.get("review") or {}
    L = [f"# Review checklist - {project.asin}", "",
         f"Video: {final_path.name}  |  length {fmt_ts(tl.total)}  |  voice: {voice_key}", ""]
    if warnings:
        L += ["## Warnings"] + [f"- {w}" for w in warnings] + [""]
    if project.compare and project.compare["kind"] == "list":
        L += ["## Products ranked (your order)"]
        for i, (sub, lab) in enumerate(zip(project.compare["products"], project.compare["labels"]), 1):
            note = project.compare["rank_labels"][i - 1] or project.compare["best_for"][i - 1]
            L.append(f"- #{i}: {lab} ({sub.asin}), {len(sub.facts.get('hands_on_notes') or [])} hands-on notes" + (f", \"{note}\"" if note else ""))
        L += ["- The ranking is presented as YOUR pick. Confirm it matches your real opinion and your own notes.", ""]
    elif project.compare:
        L += ["## Products compared"]
        for i, (sub, lab) in enumerate(zip(project.compare["products"], project.compare["labels"])):
            L.append(f"- {'ABC'[i]}: {lab} ({sub.asin}), {len(sub.facts.get('hands_on_notes') or [])} hands-on notes")
        L += [f"- Winner/preference allowed: {'yes (every product has 3+ hands-on notes)' if project.compare['pick_ok'] else 'NO: the script stays neutral'}", ""]
    L += ["## Footage", *[f"- {n}" for n in cov_notes]]
    for e in entries:
        if not e["usable"]:
            L.append(f"- EXCLUDED {e['file']}: {e['reason']}")
    L.append("")
    if rv.get("fact_issues") or rv.get("policy") or rv.get("similarity_note"):
        L += ["## Script items that still need your eyes"]
        for i in rv.get("fact_issues", []):
            L.append(f"- line [{i.get('id')}] fact check: {i.get('problem')}")
        for i in rv.get("policy", []):
            L.append(f"- line [{i['id']}] policy: {', '.join(i['problems'])}")
        if rv.get("similarity_note"):
            L.append(f"- {rv['similarity_note']}")
        L.append("")
    L += ["## Before you post",
          "- [ ] Watch the whole video with sound. Does every product claim match the listing / your own experience?",
          "- [ ] Nothing on screen shows a face, a price, a star rating or Amazon page UI.",
          *(["- [ ] Every claim is attributed to the RIGHT product" + (", and every ranking reason matches your own notes and criteria." if project.compare["kind"] == "list" else ", and every comparison point is true for ALL the products."),
             "- [ ] Each product's own link is in the description, next to its name."] if project.compare else []),
          *{"full": ["- [ ] The spoken and on-screen disclosure are in the video; paste description.txt (its first line is the disclosure) and title.txt."],
            "on_screen": ["- [ ] The disclosure appears as text on screen near the start (no voice-over line). Paste description.txt and title.txt; post pinned_comment.txt as a pinned comment."],
            "description_only": ["- [ ] Disclosure mode is description_only: nothing is said or shown inside the video. Paste description.txt (disclosure on its FIRST line) and title.txt, then post pinned_comment.txt as the first pinned comment.",
                                 "- [ ] Reminder: the FTC says a disclosure inside the video is clearer than one in the description box. If you ever want text on screen without a voice line, set disclosure: mode: on_screen in config.yaml."]}.get(disclosure_mode, []),
          "- [ ] Your affiliate link is an Amazon link (amzn.to / amazon.* with your tag), and this channel is listed in your Associates account.",
          "- [ ] YouTube's upload form asks about altered/synthetic content. Answer honestly for your AI voice.",
          "- [ ] This video differs meaningfully from your other uploads (different angle, your own footage/observations).",
          "- [ ] If the music file is yours to use, keep its licence info."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
