# Project Summary — বাংলা স্বাস্থ্য সহকারী

## What It Is

A voice-powered AI medical intake assistant that conducts structured patient interviews entirely in Bangla. Patients speak into a browser mic, the AI asks focused follow-up questions, and doctors see all collected cases on a live dashboard.

---

## What Was Built (This Session)

### Core Pipeline
- **STT**: ElevenLabs Scribe (primary, best colloquial Bangla) → Groq Whisper `large-v3` fallback
- **LLM**: Google Gemini 2.5 Flash (primary) → Groq Llama-3.3-70b fallback
- **TTS**: edge-tts `bn-BD-NabanitaNeural` (Microsoft neural Bangla voice, free)
- **RAG**: ChromaDB + `paraphrase-multilingual-MiniLM-L12-v2` embeddings

### RAG Medical Knowledge Base (`rag/knowledge.py`)
46 medical conditions covering:
- Infectious: Dengue, Malaria, Typhoid, Chikungunya, COVID-19, Cholera, TB
- Cardiac: Heart attack, Hypertension, Palpitation
- Respiratory: Pneumonia, Asthma, COPD
- GI: Gastritis, Appendicitis, Jaundice/Hepatitis, Diarrhea, Gallstones
- Neurological: Migraine, Stroke, Vertigo, Seizure
- Endocrine: Diabetes, Hypoglycemia, Thyroid
- Musculoskeletal: Arthritis, Back pain
- Urological: UTI, Kidney stones
- Women's health: Menstrual disorders, Pregnancy complications
- Mental health: Anxiety, Depression
- Paediatric: Childhood fever/rash, Malnutrition
- General: Anaemia, Oedema, Unexplained weight loss, Skin infections, Eye problems, ENT

Each entry includes: symptoms in Bangla, red flag signs, follow-up question templates, recommended tests, severity level, and clinical context.

### Conversation Flow
```
Turn 1 → Name & age (RAG off)
Turn 2 → Main symptom (RAG activates)
Turn 3 → Duration & severity
Turn 4 → Tests done → wrap-up → case sent to dashboard
```
- Max 3 sentences per response
- One question per turn
- Directives steer toward wrap-up automatically

### Doctor Dashboard (`static/dashboard.html`)
- Dark UI with sidebar navigation
- Stats row: Total / Severe / Moderate / Mild patient counts
- Filter tabs by severity
- Search by patient name or complaint
- Patient cards with color-coded avatars, severity badges, symptom chips, condition tags
- Chat-bubble style conversation transcripts
- Auto-refreshes every 5 seconds

### Persistent Storage (`pipeline/store.py`)
- SQLite at `data/cases.db` — survives restarts
- Schema: session_id, name, age, chief complaint, symptoms, duration, severity, tests done, possible conditions, summary, full transcript, turn count, timestamps
- `visit_count` column for returning patient tracking

---

## Two Versions

| | Cloud (`s2s-chatbot/`) | Local (`voice-assistant/`) |
|---|---|---|
| STT | ElevenLabs Scribe → Groq Whisper | ElevenLabs Scribe → Groq Whisper → MLX local |
| LLM | Gemini 2.5 Flash → Groq Llama | Qwen2.5 14B via llama-server (local GPU) |
| TTS | edge-tts bn-BD | edge-tts bn-BD |
| RAG | ChromaDB (same knowledge base) | ChromaDB (same knowledge base) |
| Deploy | Railway (Docker) | Local — `python app/main.py` |

---

## API Keys Required

| Key | Service | Used for |
|---|---|---|
| `ELEVENLABS_API_KEY` | elevenlabs.io (free) | Primary STT |
| `GROQ_API_KEY` | groq.com (free) | STT fallback + LLM fallback |
| `GEMINI_API_KEY` | aistudio.google.com (free) | Primary LLM (cloud version) |

---

## Fixes Made During Session

| Problem | Fix |
|---|---|
| `BeamSearchScorer` import error | Pinned `transformers==4.40.2` |
| PyTorch 2.6 `weights_only` error | Patched `torch.load` before TTS loads |
| `--flash-attn` flag syntax | Changed to `--flash-attn auto` |
| `--chat-template chatml` breaking Qwen2.5 | Removed override, let model use built-in template |
| XTTS producing garbled Bangla | Replaced with edge-tts `bn-BD-NabanitaNeural` |
| STT accuracy | Added ElevenLabs Scribe as primary |
| LLM 400 error (context overflow) | Increased llama-server ctx-size 2048→4096 |
| LLM responses cut off mid-sentence | Increased max_tokens 120→180 |
| RAG interfering with name/age turn | RAG now skips turn 1 (activates from turn 2) |
| `.claude/` skills committed to repo | Removed from git, added to `.gitignore` |

---

## Repo

**GitHub**: https://github.com/KaijokuOni/Infinity-x-Chatbot  
**Branch**: `001-bangla-voice-assistant` (ahead of `main` with all new features)
