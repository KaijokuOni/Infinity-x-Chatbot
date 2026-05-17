"""OmniCare Bangla Speech-to-Speech Chatbot — FastAPI backend."""
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from core.llm import get_llm_response
from core.medical import INTRO_MESSAGE, extract_patient_summary
from core.session import SessionManager
from core.stt import transcribe_audio
from core.tts import AUDIO_DIR, synthesize_speech

app = FastAPI(title="OmniCare Bangla S2S Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")

session_manager = SessionManager()


@app.get("/")
async def root():
    return FileResponse(str(static_dir / "index.html"))


@app.post("/api/session/start")
async def start_session():
    """Create a new session and return the opening greeting audio."""
    session_id = str(uuid.uuid4())
    session_manager.get_or_create(session_id)

    audio_path = await synthesize_speech(INTRO_MESSAGE, session_id)

    return {
        "session_id": session_id,
        "message": INTRO_MESSAGE,
        "audio_url": f"/audio/{session_id}_response.mp3",
        "stage": "intro",
    }


@app.post("/api/voice/process")
async def process_voice(audio: UploadFile = File(...), session_id: str = None):
    """Receive audio blob → transcribe → LLM → TTS → return response."""
    if not session_id:
        raise HTTPException(400, "session_id is required")

    suffix = ".webm"
    content_type = audio.content_type or ""
    if "ogg" in content_type:
        suffix = ".ogg"
    elif "wav" in content_type:
        suffix = ".wav"
    elif "mp4" in content_type or "m4a" in content_type:
        suffix = ".mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        transcription = await transcribe_audio(tmp_path)
    except ValueError as e:
        os.unlink(tmp_path)
        # Whisper couldn't make sense of audio — return a friendly retry prompt
        retry_msg = "কথা স্পষ্ট শোনা যায়নি। একটু জোরে এবং স্পষ্টভাবে আবার বলুন।"
        audio_path = await synthesize_speech(retry_msg, session_id)
        return {
            "session_id": session_id,
            "transcription": "",
            "response": retry_msg,
            "audio_url": f"/audio/{session_id}_response.mp3",
            "stage": session_manager.get_or_create(session_id)["stage"],
            "has_summary": False,
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    session = session_manager.get_or_create(session_id)
    response_text = await get_llm_response(session, transcription)
    await synthesize_speech(response_text, session_id)

    return {
        "session_id": session_id,
        "transcription": transcription,
        "response": response_text,
        "audio_url": f"/audio/{session_id}_response.mp3",
        "stage": session["stage"],
        "has_summary": session["summary"] is not None,
    }


@app.post("/api/text/process")
async def process_text(body: dict):
    """Text-only endpoint for testing without a microphone."""
    session_id = body.get("session_id") or str(uuid.uuid4())
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(400, "text field is required")

    session = session_manager.get_or_create(session_id)
    response_text = await get_llm_response(session, text)
    await synthesize_speech(response_text, session_id)

    return {
        "session_id": session_id,
        "response": response_text,
        "audio_url": f"/audio/{session_id}_response.mp3",
        "stage": session["stage"],
        "has_summary": session["summary"] is not None,
    }


@app.get("/api/session/{session_id}/summary")
async def get_summary(session_id: str):
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    summary = await extract_patient_summary(session)
    return summary


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    session_manager.clear(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/api/sessions")
async def list_sessions():
    return session_manager.list_sessions()


@app.get("/api/config")
async def get_config():
    return {
        "llm_provider": settings.llm_provider,
        "ollama_model": settings.ollama_model,
        "ollama_url": settings.ollama_url,
        "stt_provider": settings.stt_provider,
        "tts_provider": settings.tts_provider,
        "tts_voice": settings.tts_voice,
    }


@app.post("/api/config")
async def update_config(body: dict):
    allowed = {"llm_provider", "ollama_model", "stt_provider", "tts_provider", "tts_voice"}
    for key in allowed:
        if key in body:
            setattr(settings, key, body[key])
    return {"status": "updated", **await get_config()}


@app.get("/api/health")
async def health():
    ollama_ok = False
    try:
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{settings.ollama_url}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "ollama_reachable": ollama_ok,
        "stt_provider": settings.stt_provider,
        "tts_provider": settings.tts_provider,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=settings.port, reload=True)
