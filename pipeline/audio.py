"""Audio helpers: decode arbitrary browser audio to a clean WAV via ffmpeg."""
from __future__ import annotations

import subprocess

STT_SAMPLE_RATE_HZ = 16000


class AudioDecodeError(RuntimeError):
    pass


def to_wav_bytes(raw: bytes, sample_rate_hz: int = STT_SAMPLE_RATE_HZ) -> bytes:
    """Convert browser audio (webm/opus, mp4, ...) to mono 16 kHz PCM WAV bytes.

    Whisper wants 16 kHz mono; ffmpeg handles every container the browser's
    MediaRecorder might produce.
    """
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-ac", "1",
            "-ar", str(sample_rate_hz),
            # Normalize loudness so quiet mic input reaches Whisper at a
            # consistent level (reduces hallucination on faint audio).
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-f", "wav",
            "pipe:1",
        ],
        input=raw,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise AudioDecodeError(proc.stderr.decode("utf-8", "replace")[:500])
    return proc.stdout
