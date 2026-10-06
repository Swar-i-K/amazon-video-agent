"""Text-to-speech with switchable voices. Every line is cached, so re-renders never re-bill you."""
from __future__ import annotations

import asyncio
import os
import time
import wave
from pathlib import Path

import numpy as np
import requests

from .common import log, run, sha1_text
from .project import Project

SR = 44100


def resolve_voice(cfg: dict, key: str | None) -> tuple[str, dict]:
    key = key or cfg["voice"]["default"]
    if key == "mock":  # offline test voice
        return key, {"provider": "mock", "voice": "mock"}
    opts = cfg["voice"]["options"]
    if key not in opts:
        raise SystemExit(f"Unknown voice '{key}'. Options: {', '.join(opts)}")
    return key, opts[key]


# ------------------------------- providers ---------------------------------
def _eleven_headers() -> dict:
    k = os.environ.get("ELEVENLABS_API_KEY")
    if not k:
        raise SystemExit("ELEVENLABS_API_KEY is not set. Pick an edge_* voice or add the key as a GitHub secret.")
    return {"xi-api-key": k}


def eleven_subscription() -> dict | None:
    try:
        r = requests.get("https://api.elevenlabs.io/v1/user/subscription", headers=_eleven_headers(), timeout=30)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None


def eleven_list_voices() -> list[dict]:
    r = requests.get("https://api.elevenlabs.io/v1/voices", headers=_eleven_headers(), timeout=30)
    r.raise_for_status()
    return [{"voice_id": v["voice_id"], "name": v["name"], "category": v.get("category"),
             "labels": v.get("labels", {})} for v in r.json().get("voices", [])]


def _synth_eleven(cfg, v, text, out: Path, prev: str, nxt: str):
    e = cfg["voice"]["elevenlabs"]
    payload = {"text": text, "model_id": e["model_id"],
               "voice_settings": {"stability": e["stability"], "similarity_boost": e["similarity_boost"],
                                  "style": e["style"], "use_speaker_boost": True}}
    if prev:
        payload["previous_text"] = prev[-400:]
    if nxt:
        payload["next_text"] = nxt[:400]
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{v['voice_id']}?output_format=mp3_44100_128"
    for attempt in range(6):
        r = requests.post(url, headers={**_eleven_headers(), "Content-Type": "application/json"}, json=payload, timeout=120)
        if r.status_code == 200:
            out.write_bytes(r.content)
            return
        if r.status_code in (400, 422) and ("previous_text" in payload or "next_text" in payload):
            payload.pop("previous_text", None)
            payload.pop("next_text", None)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(3 + attempt * 4)
            continue
        raise RuntimeError(f"ElevenLabs error {r.status_code}: {r.text[:400]}")
    raise RuntimeError("ElevenLabs kept failing (rate limit or outage). Re-run; finished lines are cached.")


def _synth_edge(cfg, v, text, out: Path, prev: str, nxt: str):
    import edge_tts

    async def go():
        await edge_tts.Communicate(text, v["voice"], rate="-4%").save(str(out))

    last = None
    for attempt in range(4):
        try:
            asyncio.run(go())
            if out.exists() and out.stat().st_size > 1000:
                return
        except Exception as e:
            last = e
        time.sleep(2 + attempt * 3)
    raise RuntimeError(f"Edge TTS failed: {last}")


def _synth_openai(cfg, v, text, out: Path, prev: str, nxt: str):
    k = os.environ.get("OPENAI_API_KEY")
    if not k:
        raise SystemExit("OPENAI_API_KEY is not set.")
    o = cfg["voice"]["openai"]
    for attempt in range(5):
        r = requests.post("https://api.openai.com/v1/audio/speech", headers={"Authorization": f"Bearer {k}"},
                          json={"model": o["model"], "voice": v["voice"], "input": text,
                                "instructions": o["instructions"], "response_format": "mp3"}, timeout=180)
        if r.status_code == 200:
            out.write_bytes(r.content)
            return
        if r.status_code in (429, 500, 502, 503):
            time.sleep(3 + attempt * 4)
            continue
        raise RuntimeError(f"OpenAI TTS error {r.status_code}: {r.text[:400]}")
    raise RuntimeError("OpenAI TTS kept failing.")


def _synth_mock(cfg, v, text, out: Path, prev: str, nxt: str):
    """Speech-shaped tone burst for tests (no network). Writes a wav."""
    n_words = max(1, len(text.split()))
    dur = n_words / 2.7
    t = np.arange(int(SR * (dur + 0.3))) / SR
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.2 * t)
    sig = 0.25 * env * np.sin(2 * np.pi * 170 * t) * (t < dur)
    _write_wav(out, (sig * 32767).astype(np.int16))


PROVIDERS = {"elevenlabs": _synth_eleven, "edge": _synth_edge, "openai": _synth_openai, "mock": _synth_mock}


# ------------------------------- audio utils -------------------------------
def _write_wav(path: Path, data: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.astype("<i2").tobytes())


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.int16)


def _to_clean_array(src: Path, cache_wav: Path) -> np.ndarray:
    if src.suffix == ".wav":
        data = _read_wav(src)
    else:
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ar", SR, "-ac", 1, "-c:a", "pcm_s16le", cache_wav])
        data = _read_wav(cache_wav)
    f = data.astype(np.float32) / 32768.0
    amp = np.abs(f)
    win = int(SR * 0.01)
    env = np.convolve(amp, np.ones(win) / win, mode="same") if len(amp) > win else amp
    idx = np.where(env > 0.004)[0]
    if len(idx):
        a, b = max(0, idx[0] - int(SR * 0.04)), min(len(f), idx[-1] + int(SR * 0.10))
        f = f[a:b]
    fade = int(SR * 0.008)
    if len(f) > 2 * fade:
        f[:fade] *= np.linspace(0, 1, fade)
        f[-fade:] *= np.linspace(1, 0, fade)
    return (f * 32767).astype(np.int16)


# ------------------------------- main entry --------------------------------
def preflight(cfg: dict, key: str, segments: list[dict]) -> dict:
    """Check credits/licence before spending anything. Returns info used for attribution."""
    _, v = resolve_voice(cfg, key)
    info = {"provider": v["provider"], "attribution": None, "warnings": []}
    chars = sum(len(s["text"]) for s in segments)
    if v["provider"] == "elevenlabs":
        sub = eleven_subscription()
        per = cfg["voice"]["elevenlabs"]["credits_per_char"]
        need = int(chars * per)
        if sub:
            left = sub.get("character_limit", 0) - sub.get("character_count", 0)
            tier = sub.get("tier", "?")
            log(f"ElevenLabs plan: {tier}; credits left {left}; this video needs ~{need}")
            if tier == "free":
                info["attribution"] = "Voice generated by ElevenLabs"
                info["warnings"].append(
                    "ElevenLabs FREE plan: output is non-commercial and requires attribution. "
                    "A monetised/affiliate channel needs a paid plan (Starter is enough). Attribution was added automatically.")
            if left < need:
                raise SystemExit(f"Not enough ElevenLabs credits: need ~{need}, have {left}. "
                                 "Use an edge_*/openai_* voice, flash model (0.5 credit/char), or upgrade.")
        else:
            info["warnings"].append("Could not read ElevenLabs plan (API key may lack user_read). "
                                    "If you are on the free plan, add the attribution 'Voice generated by ElevenLabs' and note it is non-commercial.")
    elif v["provider"] == "edge":
        info["warnings"].append("Edge TTS uses Microsoft's unofficial free endpoint; there is no commercial licence statement. "
                                "Fine for drafts/previews; use a paid provider for monetised videos if you want certainty.")
    return info


def synthesize_all(project: Project, cfg: dict, key: str, segments: list[dict]) -> list[np.ndarray]:
    _, v = resolve_voice(cfg, key)
    synth = PROVIDERS[v["provider"]]
    tdir = project.cache_dir / "tts"
    tdir.mkdir(exist_ok=True)
    out = []
    for i, s in enumerate(segments):
        ident = sha1_text(v["provider"], v.get("voice_id", v.get("voice")), cfg["voice"].get(v["provider"], {}), s["text"])
        raw = tdir / (ident + (".wav" if v["provider"] == "mock" else ".mp3"))
        if not raw.exists():
            prev = segments[i - 1]["text"] if i else ""
            nxt = segments[i + 1]["text"] if i + 1 < len(segments) else ""
            log(f"  voice {i + 1}/{len(segments)}")
            synth(cfg, v, s["text"], raw, prev, nxt)
        out.append(_to_clean_array(raw, tdir / (ident + ".clean.wav")))
    return out
