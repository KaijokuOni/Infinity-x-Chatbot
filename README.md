# বাংলা স্বাস্থ্য সহকারী — Bangla AI Health Assistant

A voice-powered medical intake assistant that conducts structured patient interviews entirely in Bangla. Patients speak naturally into a browser mic and receive spoken Bangla replies in real time. A password-protected doctor dashboard displays all collected patient cases.

---

## Features

- **Full voice loop** — speak Bangla, hear a Bangla reply in seconds
- **Structured intake** — collects name, age, symptoms, duration, severity and tests done in 3–4 turns
- **RAG-powered questions** — 46-condition medical knowledge base guides the AI to ask clinically relevant follow-up questions
- **Doctor dashboard** — dark UI with patient cards, severity filters, search, stats and full conversation transcripts
- **Persistent storage** — all patient cases saved to SQLite, survive server restarts
- **Dual STT** — ElevenLabs Scribe (primary, best colloquial Bangla) → Groq Whisper fallback
- **Dual LLM** — Gemini 2.5 Flash (primary) → Groq Llama-3.3-70b fallback

---

## Stack

| Layer | Service |
|---|---|
| Speech-to-text (primary) | ElevenLabs Scribe `scribe_v1` — best colloquial Bangla accuracy |
| Speech-to-text (fallback) | Groq Whisper `large-v3` — used if Scribe fails |
| LLM | Google Gemini 2.5 Flash → Groq Llama-3.3-70b fallback |
| RAG | ChromaDB + `paraphrase-multilingual-MiniLM-L12-v2` (46 Bangla medical conditions) |
| Text-to-speech | edge-tts `bn-BD-NabanitaNeural` (free, no key needed) |
| Backend | FastAPI + uvicorn |
| Database | SQLite (persistent patient case storage) |

---

## Setup

### 1. API Keys

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=gsk_...            # groq.com — free tier, used for STT fallback + LLM fallback
GEMINI_API_KEY=AIza...          # aistudio.google.com — free tier, primary LLM
ELEVENLABS_API_KEY=sk_...       # elevenlabs.io — free account, primary STT (best Bangla accuracy)

# Dashboard auth (change these)
DASHBOARD_USER=doctor
DASHBOARD_PASSWORD=yourpassword

# Optional
# TTS_VOICE=bn-BD-PradeepNeural   (male voice)
```

### 2. Install dependencies

```bash
pip install uv
uv sync
```

### 3. System requirements

```bash
brew install ffmpeg          # macOS
# or: sudo apt install ffmpeg  (Ubuntu)
```

---

## Run

```bash
./run.sh          # starts server + opens ngrok public URL
./run.sh --local  # localhost only (http://localhost:8000)
```

Open the printed URL, allow microphone access, then press **"কথা বলা শুরু করুন"** and speak Bangla.

**Doctor dashboard:** `<url>/dashboard` — login with your `DASHBOARD_USER` / `DASHBOARD_PASSWORD`.

---

## Deploy (Railway)

The repo includes a `Dockerfile` and `railway.json`. Connect your Railway project to this repo and set the environment variables in the Railway dashboard. It deploys automatically on push to `main`.

---

## Project Layout

```
app.py                      FastAPI server — routes, session management, orchestration
pipeline/
  stt.py                    ElevenLabs Scribe + Groq Whisper STT
  llm.py                    Gemini 2.5 Flash + Groq fallback LLM with RAG injection
  tts.py                    edge-tts bn-BD synthesis
  audio.py                  ffmpeg: browser audio → 16 kHz mono WAV
  store.py                  SQLite persistence for patient cases
rag/
  knowledge.py              46 Bangla medical conditions (symptoms, red flags, tests, questions)
  store.py                  ChromaDB vector store + retrieval
static/
  index.html                Patient-facing voice UI
  dashboard.html            Doctor dashboard (dark UI, stats, filters, search)
data/
  cases.db                  Persistent SQLite database (auto-created)
  rag_db/                   ChromaDB vector index (auto-created on first run)
```

---

## How It Works

```
Patient speaks
    → ElevenLabs Scribe transcribes Bangla (~1s) [Groq Whisper if Scribe fails]
    → RAG retrieves relevant conditions from knowledge base
    → Gemini 2.5 Flash generates a focused follow-up question
    → edge-tts synthesizes the reply in Bangla
    → Audio plays back in browser
    → Case saved to SQLite → appears on doctor dashboard
```

The conversation ends in 4 turns:
1. Name & age
2. Main symptom
3. Duration & severity (RAG active)
4. Tests done → wrap-up summary sent to dashboard
