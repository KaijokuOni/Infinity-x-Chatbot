# বাংলা ভয়েস চ্যাটবট — Bangla Speech-to-Speech Chatbot

Hold a button, speak Bangla, hear a Bangla reply. Web-based, runs in any
browser, and can be shared over the internet with ngrok.

## Stack (all online, free-tier)

| Stage | Service |
|-------|---------|
| Speech-to-text | Groq Whisper `large-v3` (`language=bn`) |
| Chat / LLM | Google Gemini 2.5 Flash (Bangla-only system prompt, 10-turn memory) |
| Text-to-speech | edge-tts `bn-BD-NabanitaNeural` (free, no key) |
| Serving | FastAPI + browser mic UI, tunnelled via ngrok |

## Setup

1. Put your keys in `.env`:
   ```dotenv
   GROQ_API_KEY=gsk_...
   GEMINI_API_KEY=AIza...
   # optional: TTS_VOICE=bn-BD-PradeepNeural   (male voice)
   ```
2. Install deps (creates `.venv`):
   ```bash
   uv sync
   ```
3. Make sure `ffmpeg` and `ngrok` are installed:
   ```bash
   brew install ffmpeg ngrok
   ngrok config add-authtoken <your-token>   # once
   ```

## Run

```bash
./run.sh          # server + public ngrok URL
./run.sh --local  # localhost only
```

Open the printed URL, allow the microphone, then **hold the mic button (or
Spacebar)**, speak Bangla, and release. (On the free ngrok plan, click
through the one-time "Visit Site" page.)

## Layout

```
app.py              FastAPI server + /api/turn orchestration
pipeline/
  audio.py          ffmpeg: browser audio -> 16 kHz mono WAV
  stt.py            Groq Whisper
  llm.py            Gemini 2.5 Flash (Bangla)
  tts.py            edge-tts bn-BD
static/index.html   browser mic UI
run.sh              uvicorn + ngrok launcher
```
