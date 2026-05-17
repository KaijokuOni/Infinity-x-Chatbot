"""Speech-to-text: supports OpenAI Whisper API, Groq Whisper, and local mlx-whisper."""
import os
import re
import httpx
from config import settings


def _validate_transcription(text: str) -> str:
    """Reject Whisper hallucinations: repeated single chars, pure symbols, empty."""
    text = text.strip()
    if not text:
        return ""
    # Detect repetitive hallucination (e.g. "৩৩৩৩৩৩" or "।।।।।")
    # If any single character makes up >60% of the string it's garbage
    if len(text) > 5:
        for ch in set(text):
            if text.count(ch) / len(text) > 0.6:
                return ""
    # Reject strings that are only digits / punctuation with no actual letters
    if not re.search(r'[ঀ-৿A-z]', text):
        return ""
    return text


async def transcribe_audio(audio_path: str) -> str:
    provider = settings.stt_provider

    if provider == "mlx":
        raw = await _transcribe_mlx(audio_path)
    elif provider == "groq":
        raw = await _transcribe_groq(audio_path)
    elif provider == "gemini":
        raw = await _transcribe_gemini(audio_path)
    else:
        raw = await _transcribe_openai(audio_path)

    validated = _validate_transcription(raw)
    if not validated:
        raise ValueError("কথা স্পষ্ট শোনা যায়নি। আবার বলুন।")
    return validated


async def _transcribe_openai(audio_path: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    with open(audio_path, "rb") as f:
        result = await client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="bn",
            response_format="text",
        )
    return result.strip() if isinstance(result, str) else result.text.strip()


async def _transcribe_groq(audio_path: str) -> str:
    # Groq exposes OpenAI-compatible Whisper endpoint
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    with open(audio_path, "rb") as f:
        result = await client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f,
            language="bn",
            response_format="text",
        )
    return result.strip() if isinstance(result, str) else result.text.strip()


async def _transcribe_gemini(audio_path: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)

    ext = audio_path.rsplit(".", 1)[-1].lower()
    mime_map = {
        "webm": "audio/webm",
        "ogg": "audio/ogg",
        "wav": "audio/wav",
        "mp4": "audio/mp4",
        "m4a": "audio/mp4",
    }
    mime_type = mime_map.get(ext, "audio/webm")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    resp = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=[
            types.Part(text="Transcribe this audio in Bengali. Output only the transcription, no explanations."),
            types.Part(inline_data=types.Blob(mime_type=mime_type, data=audio_bytes)),
        ],
    )
    return resp.text.strip()


async def _transcribe_mlx(audio_path: str) -> str:
    try:
        import mlx_whisper  # type: ignore

        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=f"mlx-community/whisper-{settings.mlx_whisper_model}-mlx",
            language="bn",
            condition_on_previous_text=False,  # reduces hallucination loops
        )
        return result["text"].strip()
    except ImportError:
        raise RuntimeError(
            "mlx-whisper is not installed. Run: pip install mlx-whisper\n"
            "Or switch STT_PROVIDER to 'openai' or 'groq' in .env"
        )
