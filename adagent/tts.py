"""Text-to-speech with switchable voices. Every line is cached, so re-renders never re-bill you."""
from __future__ import annotations

import asyncio
import base64
import json
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
    """Best-effort Edge TTS with a fresh WebSocket communicator on every retry."""
    import edge_tts

    voice = v.get("voice")
    rate = v.get("rate", "-4%")
    volume = v.get("volume", "+0%")
    last_error = None

    for attempt in range(5):
        out.unlink(missing_ok=True)
        try:
            communicator = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate,
                volume=volume,
            )
            asyncio.run(communicator.save(str(out)))
            if _is_valid_audio(out):
                return
            last_error = RuntimeError("Edge TTS returned no audio or an undecodable audio file.")
        except Exception as exc:
            last_error = exc
        out.unlink(missing_ok=True)
        if attempt < 4:
            time.sleep(min(20, 2 ** (attempt + 1)))

    raise RuntimeError(
        f"Edge TTS failed after 5 attempts for voice {voice!r}: {last_error}. "
        "The Edge WebSocket service may be temporarily unavailable or blocked from this runner."
    )

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



def _synth_openrouter(cfg, v, text, out: Path, prev: str, nxt: str):
    """Generate TTS audio through OpenRouter. Paid use is deliberately opt-in."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set. Add it as a GitHub repository secret.")

    settings = cfg["voice"].get("openrouter", {})
    if not settings.get("allow_paid", False):
        raise SystemExit(
            "OpenRouter TTS is disabled in config.yaml. Set voice.openrouter.allow_paid: true "
            "only after checking the selected model's current price and commercial-use terms."
        )
    if os.environ.get("OPENROUTER_ALLOW_PAID_TTS", "").lower() != "true":
        raise SystemExit(
            "OpenRouter TTS requires explicit per-run approval. Set the GitHub Actions "
            "allow_paid_tts input to true for this run."
        )
    if not settings.get("commercial_use_confirmed", False):
        raise SystemExit(
            "OpenRouter TTS blocked: set voice.openrouter.commercial_use_confirmed: true "
            "only after verifying commercial-use rights for this exact model/provider."
        )

    model = v.get("model_id") or settings.get("model")
    voice_id = v.get("voice") or settings.get("voice")
    if not model or not voice_id:
        raise RuntimeError("OpenRouter voice requires model_id and voice in config.yaml.")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    referer = settings.get("http_referer")
    title = settings.get("app_title", "Summer Section Video Agent")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    payload = {
        "model": model,
        "input": text,
        "voice": voice_id,
        "response_format": settings.get("response_format", "mp3"),
    }

    last_error = None
    for attempt in range(5):
        try:
            response = requests.post(
                settings.get("endpoint", "https://openrouter.ai/api/v1/audio/speech"),
                headers=headers,
                json=payload,
                timeout=180,
            )
            if response.status_code == 200 and response.content:
                out.write_bytes(response.content)
                if _is_valid_audio(out):
                    return
                out.unlink(missing_ok=True)
                raise RuntimeError("OpenRouter returned audio that FFmpeg could not decode.")
            if response.status_code in (408, 409, 425, 429, 500, 502, 503, 504):
                last_error = RuntimeError(
                    f"OpenRouter temporary TTS error {response.status_code}: {response.text[:300]}"
                )
                time.sleep(3 + attempt * 4)
                continue
            raise RuntimeError(
                f"OpenRouter TTS error {response.status_code}: {response.text[:500]}"
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(2 + attempt * 3)

    raise RuntimeError(f"OpenRouter TTS failed after 5 attempts: {last_error}")


def _synth_inworld(cfg, v, text, out: Path, prev: str, nxt: str):
    """Generate speech with Inworld; enforce the configured lifetime usage guard."""
    settings = cfg["voice"].get("inworld", {})
    key = os.environ.get("INWORLD_API_KEY")
    if not key:
        raise SystemExit("INWORLD_API_KEY is not set. Add your Inworld API credential to GitHub Secrets.")

    voice_id = v.get("voice_id") or v.get("voice")
    model_id = v.get("model_id") or settings.get("model_id", "inworld-tts-2")
    max_units = int(settings.get("max_utf16_units", 1900))
    requested = len(text.encode("utf-16-le")) // 2
    if requested > max_units:
        raise RuntimeError(
            f"Inworld segment is {requested} UTF-16 units; configured safe maximum is {max_units}. "
            "Shorten/split this narration segment and rerun."
        )

    ledger_path = Path(settings.get("usage_ledger", ".adagent_state/tts_usage.json"))
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    except Exception as exc:
        raise RuntimeError(f"Could not read Inworld usage ledger at {ledger_path}: {exc}") from exc

    used = int(ledger.get("inworld_characters_used", 0))
    budget = int(settings.get("free_character_budget", 40000))
    paid_allowed = (
        bool(settings.get("allow_paid_fallback", False))
        and os.environ.get("INWORLD_ALLOW_PAID_TTS", "").lower() == "true"
    )
    if used + requested > budget and not paid_allowed:
        raise RuntimeError(
            f"Inworld safety budget reached: {used} characters used + {requested} requested "
            f"> {budget}. This is a conservative local lifetime cap, not a provider quota estimate. "
            "No paid fallback is enabled."
        )

    payload = {
        "text": text,
        "voiceId": voice_id,
        "modelId": model_id,
        "audioConfig": {"audioEncoding": settings.get("audio_encoding", "MP3")},
    }
    headers = {
        "Authorization": f"Basic {key}",
        "Content-Type": "application/json",
    }
    endpoint = settings.get("endpoint", "https://api.inworld.ai/tts/v1/voice")
    last_error = None

    for attempt in range(5):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=180)
            if response.status_code == 200:
                data = response.json()
                audio_content = data.get("audioContent")
                if not audio_content:
                    raise RuntimeError(f"Inworld response contained no audioContent: {str(data)[:400]}")
                out.write_bytes(base64.b64decode(audio_content))
                if not _is_valid_audio(out):
                    out.unlink(missing_ok=True)
                    raise RuntimeError("Inworld returned audio that FFmpeg could not decode.")

                actual = int(data.get("usage", {}).get("processedCharactersCount", requested))
                ledger["inworld_characters_used"] = used + actual
                ledger["inworld_model_id"] = model_id
                ledger["inworld_voice_id"] = voice_id
                ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
                return

            if response.status_code in (408, 409, 425, 429, 500, 502, 503, 504):
                last_error = RuntimeError(
                    f"Inworld temporary TTS error {response.status_code}: {response.text[:300]}"
                )
                if attempt < 4:
                    time.sleep(3 + attempt * 4)
                    continue
            else:
                raise RuntimeError(f"Inworld TTS error {response.status_code}: {response.text[:500]}")
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(2 + attempt * 3)

    raise RuntimeError(f"Inworld TTS failed after 5 attempts: {last_error}")

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
    "openrouter": _synth_openrouter,
    "inworld": _synth_inworld,
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

    elif voice["provider"] == "inworld":
        settings = cfg["voice"].get("inworld", {})
        if not settings.get("enabled", True):
            raise SystemExit("Inworld TTS is disabled in config.yaml.")
        if not os.environ.get("INWORLD_API_KEY"):
            raise SystemExit("INWORLD_API_KEY is not set. Add it as a GitHub repository secret.")
        too_long = [
            len(segment["text"].encode("utf-16-le")) // 2
            for segment in segments
            if len(segment["text"].encode("utf-16-le")) // 2 > int(settings.get("max_utf16_units", 1900))
        ]
        if too_long:
            raise SystemExit(
                f"Inworld supports at most {settings.get('max_utf16_units', 1900)} configured UTF-16 units "
                f"per request; found a segment of {max(too_long)}. Shorten the script segment and rerun."
            )
        ledger_path = Path(settings.get("usage_ledger", ".adagent_state/tts_usage.json"))
        ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
        used = int(ledger.get("inworld_characters_used", 0))
        budget = int(settings.get("free_character_budget", 40000))
        if used + chars > budget and not (
            settings.get("allow_paid_fallback", False)
            and os.environ.get("INWORLD_ALLOW_PAID_TTS", "").lower() == "true"
        ):
            raise SystemExit(
                f"Inworld safety budget would be exceeded ({used} used + {chars} needed > {budget}). "
                "No paid fallback is enabled. The configured budget is a conservative local lifetime cap."
            )
        info["warnings"].append(
            "Inworld usage is tracked in .adagent_state/tts_usage.json. The configured character cap is a "
            "conservative local safety limit, not a guaranteed provider free quota. Check your account's "
            "remaining included allowance and commercial-use terms."
        )

    elif voice["provider"] == "openrouter":
        settings = cfg["voice"].get("openrouter", {})
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise SystemExit("OPENROUTER_API_KEY is not set.")
        if not settings.get("allow_paid", False):
            raise SystemExit("OpenRouter is disabled. Set voice.openrouter.allow_paid: true to enable it.")
        if os.environ.get("OPENROUTER_ALLOW_PAID_TTS", "").lower() != "true":
            raise SystemExit("OpenRouter requires per-run approval: set allow_paid_tts=true in GitHub Actions.")
        if not settings.get("commercial_use_confirmed", False):
            raise SystemExit(
                "Commercial-use confirmation is missing. Verify the exact model/provider terms, "
                "then set voice.openrouter.commercial_use_confirmed: true."
            )
        info["warnings"].append(
            "OpenRouter is a usage-billed API gateway, not a free TTS plan. "
            "Check the selected model's current pricing and commercial-use terms before running. "
            "No automatic paid-provider fallback is configured."
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
            voice.get("model_id"),
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
                try:
                    synth(cfg, voice, segment["text"], temp_raw, previous, following)
                except Exception as primary_error:
                    fallback_cfg = cfg["voice"].get("fallback", {})
                    fallback_providers = fallback_cfg.get("providers", [])
                    can_fallback = (
                        fallback_cfg.get("enabled", False)
                        and voice["provider"] in fallback_cfg.get("only_for", ["edge"])
                    )
                    if not can_fallback or not fallback_providers:
                        raise

                    fallback_error = None
                    for provider_name in fallback_providers:
                        if provider_name not in PROVIDERS or provider_name == voice["provider"]:
                            continue
                        # Resolve a dedicated fallback voice, never reuse the failed provider's voice ID.
                        fallback_voice_key = fallback_cfg.get("voice_keys", {}).get(provider_name)
                        if not fallback_voice_key:
                            continue
                        fallback_voice = cfg["voice"]["options"].get(fallback_voice_key)
                        if not fallback_voice or fallback_voice.get("provider") != provider_name:
                            continue
                        candidate = dict(fallback_voice)
                        # Never auto-fallback to usage-billed providers.
                        if provider_name == "openrouter":
                            continue
                        fallback_temp = temp_raw.with_name(temp_raw.stem + f".{provider_name}" + temp_raw.suffix)
                        fallback_temp.unlink(missing_ok=True)
                        try:
                            log(f"  primary {voice['provider']} failed; trying approved fallback {provider_name}")
                            PROVIDERS[provider_name](cfg, candidate, segment["text"], fallback_temp, previous, following)
                            if not _is_valid_audio(fallback_temp):
                                raise RuntimeError(f"{provider_name} fallback returned invalid audio.")
                            fallback_temp.replace(temp_raw)
                            fallback_error = None
                            break
                        except Exception as exc:
                            fallback_error = exc
                            fallback_temp.unlink(missing_ok=True)

                    if fallback_error is not None:
                        raise RuntimeError(
                            f"Primary {voice['provider']} TTS failed ({primary_error}); "
                            f"configured fallback(s) also failed ({fallback_error})."
                        ) from primary_error
                    if not temp_raw.exists():
                        raise RuntimeError(
                            f"Primary {voice['provider']} TTS failed ({primary_error}); "
                            "no configured fallback produced audio."
                        ) from primary_error

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
