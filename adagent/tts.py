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
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise SystemExit(
            "ELEVENLABS_API_KEY is not set. Pick an edge_* voice or add the key as a GitHub secret."
        )
    return {"xi-api-key": key, "Content-Type": "application/json"}


def eleven_subscription() -> dict | None:
    try:
        response = requests.get(
            "https://api.elevenlabs.io/v1/user/subscription",
            headers=_eleven_headers(),
            timeout=30,
        )
        return response.json() if response.status_code == 200 else None
    except requests.RequestException:
        return None


def eleven_list_voices() -> list[dict]:
    response = requests.get(
        "https://api.elevenlabs.io/v1/voices",
        headers=_eleven_headers(),
        timeout=30,
    )
    response.raise_for_status()
    return [
        {
            "voice_id": voice["voice_id"],
            "name": voice["name"],
            "category": voice.get("category"),
            "labels": voice.get("labels", {}),
        }
        for voice in response.json().get("voices", [])
    ]


def _synth_eleven(cfg, v, text, out: Path, prev: str, nxt: str):
    """Generate ElevenLabs MP3 audio, retrying transient failures."""
    settings = cfg["voice"].get("elevenlabs", {})
    voice_id = v.get("voice_id")
    if not voice_id:
        raise RuntimeError("The selected ElevenLabs voice is missing voice_id.")

    model_id = settings.get("model_id", settings.get("model", "eleven_multilingual_v2"))
    payload = {"text": text, "model_id": model_id}

    # Preserve configured voice settings when present.
    voice_settings = settings.get("voice_settings")
    if voice_settings:
        payload["voice_settings"] = voice_settings

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    last_error = None

    for attempt in range(5):
        try:
            response = requests.post(
                url,
                headers=_eleven_headers(),
                params={"output_format": settings.get("output_format", "mp3_44100_128")},
                json=payload,
                timeout=180,
            )
            if response.status_code == 200 and response.content:
                out.write_bytes(response.content)
                if _is_valid_audio(out):
                    return
                out.unlink(missing_ok=True)
                last_error = RuntimeError("ElevenLabs returned audio that FFmpeg could not decode.")
            elif response.status_code in (429, 500, 502, 503, 504):
                last_error = RuntimeError(
                    f"ElevenLabs temporary error {response.status_code}: {response.text[:300]}"
                )
                time.sleep(3 + attempt * 4)
                continue
            else:
                raise RuntimeError(
                    f"ElevenLabs TTS error {response.status_code}: {response.text[:400]}"
                )
        except requests.RequestException as exc:
            last_error = exc

        if attempt < 4:
            time.sleep(2 + attempt * 3)

    raise RuntimeError(f"ElevenLabs TTS failed after 5 attempts: {last_error}")


def _synth_edge(cfg, v, text, out: Path, prev: str, nxt: str):
    """Generate Edge TTS audio and retry if the resulting file is invalid."""
    import edge_tts

    async def go():
        await edge_tts.Communicate(
            text,
            v["voice"],
            rate="-4%",
        ).save(str(out))

    last_error = None
    for attempt in range(4):
        out.unlink(missing_ok=True)
        try:
            asyncio.run(go())
            if _is_valid_audio(out):
                return
            last_error = RuntimeError("Edge TTS returned invalid or incomplete audio.")
        except Exception as exc:
            last_error = exc

        out.unlink(missing_ok=True)
        if attempt < 3:
            time.sleep(2 + attempt * 3)

    raise RuntimeError(f"Edge TTS failed after 4 attempts: {last_error}")


def _synth_openai(cfg, v, text, out: Path, prev: str, nxt: str):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    settings = cfg["voice"].get("openai", {})
    for attempt in range(5):
        response = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": settings["model"],
                "voice": v["voice"],
                "input": text,
                "instructions": settings.get("instructions", ""),
                "response_format": "mp3",
            },
            timeout=180,
        )
        if response.status_code == 200:
            out.write_bytes(response.content)
            if _is_valid_audio(out):
                return
            out.unlink(missing_ok=True)
            raise RuntimeError("OpenAI TTS returned audio that FFmpeg could not decode.")

        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(3 + attempt * 4)
            continue
        raise RuntimeError(f"OpenAI TTS error {response.status_code}: {response.text[:400]}")

    raise RuntimeError("OpenAI TTS kept failing after 5 attempts.")


def _synth_mock(cfg, v, text, out: Path, prev: str, nxt: str):
    """Speech-shaped tone burst for tests (no network). Writes a WAV."""
    n_words = max(1, len(text.split()))
    duration = n_words / 2.7
    t = np.arange(int(SR * (duration + 0.3))) / SR
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.2 * t)
    signal = 0.25 * envelope * np.sin(2 * np.pi * 170 * t) * (t < duration)
    _write_wav(out, (signal * 32767).astype(np.int16))


# Keep this registry after all provider functions have been defined.
PROVIDERS = {
    "elevenlabs": _synth_eleven,
    "edge": _synth_edge,
    "openai": _synth_openai,
    "mock": _synth_mock,
}


# ------------------------------- audio utils -------------------------------
def _is_valid_audio(path: Path) -> bool:
    """Return True only when FFmpeg can decode at least one audio stream."""
    try:
        if not path.is_file() or path.stat().st_size < 44:
            return False
        run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-f",
                "null",
                "-",
            ]
        )
        return True
    except Exception:
        return False


def _write_wav(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SR)
        wav_file.writeframes(data.astype("<i2").tobytes())


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        return np.frombuffer(
            wav_file.readframes(wav_file.getnframes()),
            dtype="<i2",
        ).astype(np.int16)


def _to_clean_array(src: Path, cache_wav: Path) -> np.ndarray:
    """Decode provider audio to mono 44.1 kHz PCM and trim silence."""
    if not _is_valid_audio(src):
        raise RuntimeError(f"Cannot decode invalid or empty TTS audio: {src.name}")

    if src.suffix.lower() == ".wav":
        data = _read_wav(src)
    else:
        temp_wav = cache_wav.with_name(cache_wav.stem + ".partial.wav")
        temp_wav.unlink(missing_ok=True)
        try:
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(src),
                    "-ar",
                    str(SR),
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(temp_wav),
                ]
            )
            if not _is_valid_audio(temp_wav):
                raise RuntimeError(f"FFmpeg produced an invalid WAV from {src.name}")
            temp_wav.replace(cache_wav)
            data = _read_wav(cache_wav)
        finally:
            temp_wav.unlink(missing_ok=True)

    if len(data) == 0:
        raise RuntimeError(f"TTS audio contains no samples: {src.name}")

    audio = data.astype(np.float32) / 32768.0
    amplitude = np.abs(audio)
    window = int(SR * 0.01)
    envelope = (
        np.convolve(amplitude, np.ones(window) / window, mode="same")
        if len(audio) > window
        else amplitude
    )
    active = np.where(envelope > 0.004)[0]
    if len(active):
        start = max(0, active[0] - int(SR * 0.04))
        end = min(len(audio), active[-1] + int(SR * 0.10))
        audio = audio[start:end]

    fade = int(SR * 0.008)
    if len(audio) > 2 * fade:
        audio[:fade] *= np.linspace(0, 1, fade)
        audio[-fade:] *= np.linspace(1, 0, fade)

    return (audio * 32767).astype(np.int16)


# ------------------------------- main entry --------------------------------
def preflight(cfg: dict, key: str, segments: list[dict]) -> dict:
    """Check credits/licence before spending anything. Returns info used for attribution."""
    _, voice = resolve_voice(cfg, key)
    info = {"provider": voice["provider"], "attribution": None, "warnings": []}
    chars = sum(len(segment["text"]) for segment in segments)

    if voice["provider"] == "elevenlabs":
        subscription = eleven_subscription()
        credits_per_char = cfg["voice"]["elevenlabs"]["credits_per_char"]
        needed = int(chars * credits_per_char)

        if subscription:
            remaining = subscription.get("character_limit", 0) - subscription.get("character_count", 0)
            tier = subscription.get("tier", "?")
            log(f"ElevenLabs plan: {tier}; credits left {remaining}; this video needs ~{needed}")

            if tier == "free":
                info["attribution"] = "Voice generated by ElevenLabs"
                info["warnings"].append(
                    "ElevenLabs FREE plan: output is non-commercial and requires attribution. "
                    "A monetised/affiliate channel needs a paid plan (Starter is enough). "
                    "Attribution was added automatically."
                )

            if remaining < needed:
                raise SystemExit(
                    f"Not enough ElevenLabs credits: need ~{needed}, have {remaining}. "
                    "Use an edge_*/openai_* voice, flash model (0.5 credit/char), or upgrade."
                )
        else:
            info["warnings"].append(
                "Could not read ElevenLabs plan (API key may lack user_read). "
                "If you are on the free plan, add the attribution 'Voice generated by ElevenLabs' "
                "and note it is non-commercial."
            )

    elif voice["provider"] == "edge":
        info["warnings"].append(
            "Edge TTS uses Microsoft's unofficial free endpoint; there is no commercial licence statement. "
            "Fine for drafts/previews; use a paid provider for monetised videos if you want certainty."
        )

    return info


def synthesize_all(
    project: Project,
    cfg: dict,
    key: str,
    segments: list[dict],
) -> list[np.ndarray]:
    _, voice = resolve_voice(cfg, key)
    synth = PROVIDERS[voice["provider"]]

    tdir = project.cache_dir / "tts"
    tdir.mkdir(parents=True, exist_ok=True)

    output = []

    for i, segment in enumerate(segments):
        ident = sha1_text(
            voice["provider"],
            voice.get("voice_id", voice.get("voice")),
            cfg["voice"].get(voice["provider"], {}),
            segment["text"],
        )

        extension = ".wav" if voice["provider"] == "mock" else ".mp3"
        raw = tdir / (ident + extension)

        # Validate existing cache before reusing it; discard corrupt cache entries.
        if raw.exists() and not _is_valid_audio(raw):
            log(f"  removing invalid cached audio: {raw.name}")
            raw.unlink(missing_ok=True)

        if not raw.exists():
            previous = segments[i - 1]["text"] if i else ""
            following = segments[i + 1]["text"] if i + 1 < len(segments) else ""

            temp_raw = tdir / (ident + ".partial" + extension)
            temp_raw.unlink(missing_ok=True)
            log(f"  voice {i + 1}/{len(segments)}")

            try:
                synth(cfg, voice, segment["text"], temp_raw, previous, following)

                if not _is_valid_audio(temp_raw):
                    raise RuntimeError(f"Invalid TTS audio for segment {i + 1}")

                # Only move a successfully decoded file into the persistent cache.
                temp_raw.replace(raw)
            except Exception:
                temp_raw.unlink(missing_ok=True)
                raise

        clean = _to_clean_array(raw, tdir / (ident + ".clean.wav"))
        output.append(clean)

    return output
