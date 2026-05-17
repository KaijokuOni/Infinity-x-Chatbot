"""Text-to-speech: edge-tts (Microsoft neural Bangla) or gTTS fallback."""
import asyncio
import os
import tempfile
from pathlib import Path

from config import settings

AUDIO_DIR = Path(__file__).parent.parent / "audio_cache"
AUDIO_DIR.mkdir(exist_ok=True)


async def synthesize_speech(text: str, session_id: str) -> Path:
    """Returns path to the generated MP3 file."""
    out_path = AUDIO_DIR / f"{session_id}_response.mp3"

    if settings.tts_provider == "gtts":
        await _synth_gtts(text, out_path)
    else:
        await _synth_edge(text, out_path)

    return out_path


async def _synth_edge(text: str, out_path: Path) -> None:
    try:
        import edge_tts  # type: ignore

        communicate = edge_tts.Communicate(text, voice=settings.tts_voice)
        await communicate.save(str(out_path))
    except ImportError:
        raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")


async def _synth_gtts(text: str, out_path: Path) -> None:
    try:
        from gtts import gTTS  # type: ignore
        import asyncio

        def _blocking():
            tts = gTTS(text=text, lang="bn")
            tts.save(str(out_path))

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _blocking)
    except ImportError:
        raise RuntimeError("gTTS not installed. Run: pip install gTTS")


async def synthesize_greeting() -> Path:
    """Pre-generate the opening greeting audio."""
    greeting = (
        "আমি ভীণা, আপনার স্বাস্থ্য সহকারী। "
        "আপনার সমস্যার কথা বলুন — আমি মনোযোগ দিয়ে শুনছি।"
    )
    return await synthesize_speech(greeting, "greeting")
