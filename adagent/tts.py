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
