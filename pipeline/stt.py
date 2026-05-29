"""Speech-to-text.

Primary: ElevenLabs Scribe (`scribe_v1`) — accurate multilingual STT incl.
Bangla. Far better on colloquial Bangla than Whisper. Needs a free ElevenLabs
API key (ELEVENLABS_API_KEY); email/Google signup, no card.
Fallback: Groq Whisper large-v3 — used when Scribe is unavailable or returns
nothing usable.
"""
from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass

import httpx
from groq import Groq

WHISPER_MODEL_ID = "whisper-large-v3"
SCRIBE_MODEL_ID = "scribe_v1"
MODEL_ID = SCRIBE_MODEL_ID  # primary, shown on /health
LANGUAGE = "bn"
SCRIBE_LANGUAGE_CODE = "bn"
SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"


class SttError(RuntimeError):
    pass


@dataclass
class SttResult:
    text: str
    latency_ms: int
    engine: str


# -------------------------------------------------------- ElevenLabs STT ----
def _scribe_available() -> bool:
    return bool(os.environ.get("ELEVENLABS_API_KEY"))


def _transcribe_scribe(wav_bytes: bytes) -> str:
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise SttError("ELEVENLABS_API_KEY is not set")
    resp = httpx.post(
        SCRIBE_URL,
        headers={"xi-api-key": key},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        data={"model_id": SCRIBE_MODEL_ID, "language_code": SCRIBE_LANGUAGE_CODE},
        timeout=30.0,
    )
    resp.raise_for_status()
    return (resp.json().get("text") or "").strip()


# ---------------------------------------------------------------- Whisper ----
_groq: Groq | None = None


def _groq_client() -> Groq:
    global _groq
    if _groq is None:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise SttError("GROQ_API_KEY is not set")
        _groq = Groq(api_key=key)
    return _groq


def _transcribe_whisper(wav_bytes: bytes) -> str:
    client = _groq_client()
    buf = io.BytesIO(wav_bytes)
    buf.name = "audio.wav"
    # No prompt: a prompt makes Whisper hallucinate that exact text back on
    # silent/empty audio (seen on devices whose mic captures no real speech).
    resp = client.audio.transcriptions.create(
        file=buf,
        model=WHISPER_MODEL_ID,
        language=LANGUAGE,
        temperature=0.0,
    )
    return (resp.text or "").strip()


# --------------------------------------------------------------- public ----
def transcribe(wav_bytes: bytes) -> SttResult:
    """Transcribe Bangla WAV audio (16 kHz mono LINEAR16) to Bangla text."""
    start = time.monotonic()

    if _scribe_available():
        try:
            text = _transcribe_scribe(wav_bytes)
            if text:
                return SttResult(
                    text=text,
                    latency_ms=int((time.monotonic() - start) * 1000),
                    engine=SCRIBE_MODEL_ID,
                )
        except Exception:  # noqa: BLE001 — fall back to Whisper
            pass

    try:
        text = _transcribe_whisper(wav_bytes)
    except Exception as exc:  # noqa: BLE001
        raise SttError(str(exc)) from exc
    return SttResult(
        text=text,
        latency_ms=int((time.monotonic() - start) * 1000),
        engine=WHISPER_MODEL_ID,
    )
