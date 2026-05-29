"""Bangla text-to-speech.

Primary: ElevenLabs (`eleven_flash_v2_5` — supports Bengali, half the credit
cost so the free monthly quota lasts longer). Needs ELEVENLABS_API_KEY.
Fallback: edge-tts (free, unlimited, dedicated bn-BD neural voice) — used when
ElevenLabs is unavailable or the monthly character quota is exhausted.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

import edge_tts
import httpx

# edge-tts bn-BD neural voices. Nabanita = female, Pradeep = male.
DEFAULT_VOICE = os.environ.get("TTS_VOICE", "bn-BD-NabanitaNeural")

# ElevenLabs: eleven_v3 is the only model that pronounces Bengali correctly
# (flash/turbo/multilingual_v2 mangle it). Default voice configurable.
ELEVEN_MODEL_ID = "eleven_v3"
ELEVEN_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
ELEVEN_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


class TtsError(RuntimeError):
    pass


@dataclass
class TtsResult:
    mp3_bytes: bytes
    mime: str
    latency_ms: int


# -------------------------------------------------------- ElevenLabs TTS ----
def _eleven_available() -> bool:
    return bool(os.environ.get("ELEVENLABS_API_KEY"))


def _synthesize_eleven(text: str) -> bytes:
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise TtsError("ELEVENLABS_API_KEY is not set")
    resp = httpx.post(
        ELEVEN_URL.format(voice_id=ELEVEN_VOICE_ID),
        headers={"xi-api-key": key, "Content-Type": "application/json"},
        json={
            "text": text,
            "model_id": ELEVEN_MODEL_ID,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        },
        params={"output_format": "mp3_44100_128"},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.content


# ------------------------------------------------------------- edge-tts ----
async def _synthesize_edge_async(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    chunks = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.extend(chunk["data"])
    return bytes(chunks)


def _synthesize_edge(text: str, voice: str) -> bytes:
    return asyncio.run(_synthesize_edge_async(text, voice))


# --------------------------------------------------------------- public ----
def synthesize(text: str, voice: str | None = None) -> TtsResult:
    """Convert Bangla text to MP3 audio bytes."""
    if not text.strip():
        raise TtsError("empty text")
    start = time.monotonic()

    if _eleven_available():
        try:
            audio = _synthesize_eleven(text)
            if audio:
                return TtsResult(
                    mp3_bytes=audio,
                    mime="audio/mpeg",
                    latency_ms=int((time.monotonic() - start) * 1000),
                )
        except Exception:  # noqa: BLE001 — fall back to edge-tts
            pass

    voice = voice or DEFAULT_VOICE
    try:
        audio = _synthesize_edge(text, voice)
    except Exception as exc:  # noqa: BLE001
        raise TtsError(str(exc)) from exc
    if not audio:
        raise TtsError("no audio produced")
    return TtsResult(
        mp3_bytes=audio,
        mime="audio/mpeg",
        latency_ms=int((time.monotonic() - start) * 1000),
    )
