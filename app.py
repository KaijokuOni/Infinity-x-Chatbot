"""Bangla speech-to-speech chatbot — FastAPI server.

Flow per turn:
    browser mic -> POST /api/turn -> ffmpeg decode -> Groq Whisper (STT)
    -> Gemini 2.5 Flash (LLM, Bangla) -> edge-tts (TTS) -> audio back to browser

Run:  python app.py    (or:  ./run.sh  to also open an ngrok tunnel)
"""
from __future__ import annotations

import base64
import os
import secrets
import threading
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from pipeline import llm, stt, store, tts
from pipeline.audio import AudioDecodeError, to_wav_bytes

load_dotenv()

MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "10"))
# Cap the interview at this many patient turns (turn 1 ≈ name/age), then wrap up.
MAX_QA_TURNS = int(os.environ.get("MAX_QA_TURNS", "6"))
RETRY_TEXT = "দুঃখিত, আমি বুঝতে পারিনি। আবার একটু বলবেন?"

# Spoken opening line — the assistant asks for name and age by voice.
GREETING_TEXT = (
    "আমি আপনার স্বাস্থ্য সহকারী। শুরু করার আগে অনুগ্রহ করে আপনার নাম "
    "এবং বয়স বলুন।"
)

# Steer the assistant toward asking about tests, then closing, near the end.
DIRECTIVE_ASK_TESTS = (
    "কথোপকথন প্রায় শেষের দিকে। এই উত্তরে রোগীকে অবশ্যই জিজ্ঞাসা করো সে এখন "
    "পর্যন্ত কোন কোন মেডিকেল পরীক্ষা বা টেস্ট করিয়েছে (যদি আগে জিজ্ঞাসা না করে থাকো)।"
)
DIRECTIVE_WRAP_UP = (
    "এটি কথোপকথনের শেষ ধাপ। আর কোনো নতুন প্রশ্ন করবে না। সংক্ষেপে রোগীর মূল "
    "উপসর্গগুলো একবার বলো, রোগীকে ধন্যবাদ জানাও, এবং বলো যে এই তথ্য এখন "
    "ডাক্তারের ড্যাশবোর্ডে পাঠানো হয়েছে।"
)

_STATIC = Path(__file__).parent / "static"

# Doctor dashboard auth. Set DASHBOARD_PASSWORD in the deploy env; the default
# only exists so the gate is never accidentally left open.
DASHBOARD_USER = os.environ.get("DASHBOARD_USER", "doctor")
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "changeme")
_security = HTTPBasic()


def require_doctor(credentials: HTTPBasicCredentials = Depends(_security)) -> None:
    """Gate the dashboard + cases API behind HTTP Basic auth."""
    user_ok = secrets.compare_digest(credentials.username, DASHBOARD_USER)
    pass_ok = secrets.compare_digest(credentials.password, DASHBOARD_PASSWORD)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )


app = FastAPI(title="Bangla S2S Chatbot")
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

store.init_db()

# In-memory conversation history per browser session_id.
_history: dict[str, list[dict]] = {}
_lock = threading.Lock()


def _update_case(session_id: str) -> None:
    """Background: extract a structured case (incl. name/age) and persist it."""
    with _lock:
        hist = list(_history.get(session_id, []))
    if not hist:
        return
    try:
        structured = llm.extract_case(hist)
    except Exception:  # noqa: BLE001 — still store the transcript
        structured = {}
    store.save_case(
        session_id, structured, hist,
        name=str(structured.get("patient_name", "")),
        age=str(structured.get("patient_age", "")),
    )


def _count_bengali(text: str) -> int:
    return sum(1 for ch in text if 0x0980 <= ord(ch) <= 0x09FF)


def _is_garbage(text: str) -> bool:
    """Reject empty / non-Bengali / too-short transcriptions before the LLM."""
    return _count_bengali(text.strip()) < 2


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((_STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(_: None = Depends(require_doctor)) -> HTMLResponse:
    return HTMLResponse((_STATIC / "dashboard.html").read_text(encoding="utf-8"))


@app.get("/api/cases")
def api_cases(_: None = Depends(require_doctor)) -> JSONResponse:
    return JSONResponse(store.list_cases())


@app.get("/api/greeting")
def greeting() -> JSONResponse:
    """Spoken opening line: the assistant asks the patient's name and age."""
    return _voice_reply(ok=True, user_text="", reply_text=GREETING_TEXT,
                        stage="greeting", latency_ms=0)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "stt": stt.SCRIBE_MODEL_ID if stt._scribe_available() else stt.WHISPER_MODEL_ID,
        "stt_fallback": stt.WHISPER_MODEL_ID,
        "llm": llm.GEMINI_MODEL_ID,
        "llm_fallback": llm.GROQ_MODEL_ID,
        "tts": tts.ELEVEN_MODEL_ID if tts._eleven_available() else tts.DEFAULT_VOICE,
        "tts_fallback": tts.DEFAULT_VOICE,
    }


@app.post("/api/turn")
def turn(audio: UploadFile, session_id: str = Form(...)) -> JSONResponse:
    # Sync route -> runs in a threadpool, so edge-tts's asyncio.run() is safe.
    raw = audio.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="খালি অডিও পাঠানো হয়েছে।")

    try:
        wav = to_wav_bytes(raw)
    except AudioDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"অডিও ডিকোড ব্যর্থ: {exc}")

    # ---- STT ----
    try:
        stt_res = stt.transcribe(wav)
    except stt.SttError as exc:
        return JSONResponse({"ok": False, "stage": "stt", "error": str(exc)})

    user_text = stt_res.text
    if _is_garbage(user_text):
        return _voice_reply(ok=False, user_text=user_text, reply_text=RETRY_TEXT,
                            stage="filter", latency_ms=stt_res.latency_ms)

    # ---- LLM (with history + stage directive) ----
    with _lock:
        hist = list(_history.get(session_id, []))

    upcoming = len(hist) + 1  # which patient turn this is (1-based)
    done = upcoming >= MAX_QA_TURNS
    if done:
        directive = DIRECTIVE_WRAP_UP
    elif upcoming == MAX_QA_TURNS - 1:
        directive = DIRECTIVE_ASK_TESTS
    else:
        directive = None

    try:
        llm_res = llm.reply(user_text, history=hist, directive=directive)
    except llm.LlmError as exc:
        return JSONResponse({"ok": False, "stage": "llm",
                             "user_text": user_text, "error": str(exc)})

    # ---- record turn ----
    with _lock:
        h = _history.setdefault(session_id, [])
        h.append({"user": user_text, "assistant": llm_res.text})
        del h[:-MAX_HISTORY]

    # Update the doctor dashboard off the request path.
    threading.Thread(target=_update_case, args=(session_id,), daemon=True).start()

    return _voice_reply(
        ok=True, user_text=user_text, reply_text=llm_res.text, stage="ok",
        latency_ms=stt_res.latency_ms + llm_res.latency_ms, model=llm_res.model,
        done=done,
    )


def _voice_reply(*, ok: bool, user_text: str, reply_text: str, stage: str,
                 latency_ms: int, model: str | None = None,
                 done: bool = False) -> JSONResponse:
    """Synthesize reply_text and package the JSON response."""
    audio_b64 = None
    mime = None
    tts_error = None
    try:
        tts_res = tts.synthesize(reply_text)
        audio_b64 = base64.b64encode(tts_res.mp3_bytes).decode("ascii")
        mime = tts_res.mime
        latency_ms += tts_res.latency_ms
    except tts.TtsError as exc:
        tts_error = str(exc)

    return JSONResponse({
        "ok": ok,
        "stage": stage,
        "user_text": user_text,
        "reply_text": reply_text,
        "reply_audio_b64": audio_b64,
        "reply_audio_mime": mime,
        "tts_error": tts_error,
        "latency_ms": latency_ms,
        "model": model,
        "done": done,
    })


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
