"""Bangla chat replies.

Primary: Google Gemini 2.5 Flash (with retry/backoff on transient 503/429).
Fallback: Groq-hosted Llama 3.3 70B when Gemini stays unavailable. Both are
online; the fallback uses the GROQ_API_KEY you already have for STT.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from google import genai
from google.genai import types

GEMINI_MODEL_ID = "gemini-2.5-flash"
GROQ_MODEL_ID = "llama-3.3-70b-versatile"
MODEL_ID = GEMINI_MODEL_ID  # primary, shown on /health
MAX_OUTPUT_TOKENS = 400

GEMINI_RETRIES = 3
BACKOFF_SECONDS = (0.6, 1.5, 3.0)

SYSTEM_PROMPT = (
    "তুমি একজন বাংলা মেডিকেল ভয়েস সহকারী। কাজ: রোগীর উপসর্গ সংগ্রহ করা। "
    "চিকিৎসা বা পরামর্শ দেবে না।\n\n"
    "নিয়ম:\n"
    "১. প্রতি উত্তরে শুধু একটি প্রশ্ন করবে।\n"
    "২. উত্তর সর্বোচ্চ ৩ বাক্যের মধ্যে রাখবে।\n"
    "৩. নাম ও বয়স পেলে শুধু স্বীকার করে সরাসরি সমস্যা জিজ্ঞাসা করো।\n"
    "৪. উপসর্গ শুনলে: কতদিন ধরে এবং তীব্রতা জিজ্ঞাসা করো।\n"
    "৫. কোনো তালিকা, বুলেট বা দীর্ঘ ব্যাখ্যা নয়।\n"
    "৬. সহানুভূতিশীল কিন্তু সংক্ষিপ্ত থাকো।\n"
    "৭. নাম ধরে ডাকবে না, 'ভাই/আপা/দাদা' বলবে না।"
)

_TRANSIENT_MARKERS = (
    "503", "unavailable", "overloaded", "high demand", "try again",
    "deadline", "timeout", "500", "internal",
)
# Quota errors aren't worth retrying — fall back immediately and stop calling
# Gemini for a while (it keeps 429-ing once the daily free tier is spent).
_QUOTA_MARKERS = ("429", "resource_exhausted", "quota", "exceeded")
GEMINI_COOLDOWN_SECONDS = 600
_gemini_cooldown_until = 0.0


class LlmError(RuntimeError):
    pass


@dataclass
class LlmResult:
    text: str
    latency_ms: int
    model: str


def _has_bengali(text: str) -> bool:
    return any(0x0980 <= ord(ch) <= 0x09FF for ch in text)


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _TRANSIENT_MARKERS)


def _is_quota(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _QUOTA_MARKERS)


# ---------------------------------------------------------------- Gemini ----
_gemini: genai.Client | None = None


def _gemini_client() -> genai.Client:
    global _gemini
    if _gemini is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise LlmError("GEMINI_API_KEY is not set")
        _gemini = genai.Client(api_key=key)
    return _gemini


def _gemini_contents(history: list[dict], user_text: str) -> list[types.Content]:
    contents: list[types.Content] = []
    for turn in history:
        contents.append(types.Content(role="user", parts=[types.Part(text=turn["user"])]))
        contents.append(types.Content(role="model", parts=[types.Part(text=turn["assistant"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
    return contents


def _build_system(rag_context: str = "", directive: str | None = None) -> str:
    parts = [SYSTEM_PROMPT]
    if rag_context:
        parts.append(rag_context)
    if directive:
        parts.append(directive)
    return "\n\n".join(parts)


def _gemini_reply(user_text: str, history: list[dict], directive: str | None = None,
                  rag_context: str = "") -> str:
    client = _gemini_client()
    system = _build_system(rag_context, directive)
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.7,
    )
    last_exc: Exception | None = None
    for attempt in range(GEMINI_RETRIES):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL_ID,
                contents=_gemini_contents(history, user_text),
                config=config,
            )
            return (resp.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            # Retry only transient blips, never quota errors.
            if _is_transient(exc) and not _is_quota(exc) and attempt < GEMINI_RETRIES - 1:
                time.sleep(BACKOFF_SECONDS[attempt])
                continue
            raise
    raise last_exc if last_exc else LlmError("Gemini failed")


# ------------------------------------------------------------------ Groq ----
_groq = None


def _groq_client():
    global _groq
    if _groq is None:
        from groq import Groq

        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise LlmError("GROQ_API_KEY is not set")
        _groq = Groq(api_key=key)
    return _groq


def _groq_reply(user_text: str, history: list[dict], directive: str | None = None,
                rag_context: str = "") -> str:
    client = _groq_client()
    system = _build_system(rag_context, directive)
    messages = [{"role": "system", "content": system}]
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": user_text})
    resp = client.chat.completions.create(
        model=GROQ_MODEL_ID,
        messages=messages,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.7,
    )
    return (resp.choices[0].message.content or "").strip()


# --------------------------------------------------------------- public ----
def reply(
    user_text: str,
    history: list[dict] | None = None,
    directive: str | None = None,
) -> LlmResult:
    """Generate a Bangla reply. `history` is a list of {"user","assistant"} dicts.

    `directive` is an optional extra instruction appended to the system prompt
    for this turn (used to steer the conversation toward wrap-up).
    """
    global _gemini_cooldown_until
    history = history or []
    start = time.monotonic()

    # RAG: inject medical context after name/age turn (turn 2+)
    rag_context = ""
    if history:
        try:
            from rag.store import build_context_block
            rag_context = build_context_block(user_text, history)
        except Exception as exc:
            pass  # RAG is optional — never block a reply

    # Skip Gemini entirely while it's in quota cooldown — avoids a wasted call
    # (and its network round-trip) on every turn once the daily tier is spent.
    skip_gemini = time.monotonic() < _gemini_cooldown_until

    model_used = GROQ_MODEL_ID if skip_gemini else GEMINI_MODEL_ID
    if skip_gemini:
        text = _groq_reply(user_text, history, directive, rag_context)
    else:
        try:
            text = _gemini_reply(user_text, history, directive, rag_context)
        except Exception as gemini_exc:  # noqa: BLE001 — try the fallback
            if _is_quota(gemini_exc):
                _gemini_cooldown_until = time.monotonic() + GEMINI_COOLDOWN_SECONDS
            try:
                text = _groq_reply(user_text, history, directive, rag_context)
                model_used = GROQ_MODEL_ID
            except Exception as groq_exc:  # noqa: BLE001
                raise LlmError(
                    f"Gemini failed ({gemini_exc}); Groq fallback failed ({groq_exc})"
                ) from groq_exc

    # Bangla-only guard: one corrective retry on whichever model answered.
    if text and not _has_bengali(text):
        nudge = user_text + "\n\n(অনুগ্রহ করে শুধু বাংলায় উত্তর দাও।)"
        try:
            text = (
                _groq_reply(nudge, history, directive, rag_context)
                if model_used == GROQ_MODEL_ID
                else _gemini_reply(nudge, history, directive, rag_context)
            )
        except Exception:  # noqa: BLE001 — keep the original text
            pass

    latency_ms = int((time.monotonic() - start) * 1000)
    if not text:
        text = "দুঃখিত, আমি এই মুহূর্তে উত্তর দিতে পারছি না।"
    return LlmResult(text=text, latency_ms=latency_ms, model=model_used)


# ------------------------------------------------ structured extraction ----
EXTRACT_SYSTEM = (
    "You are a medical scribe. From the Bangla doctor-intake conversation below, "
    "extract a structured case for a doctor. Return ONLY a JSON object with keys: "
    '"patient_name" (the patient\'s name as stated, or ""), '
    '"patient_age" (the patient\'s age as stated, digits only as a string, or ""), '
    '"chief_complaint" (short Bangla string), '
    '"symptoms" (array of short Bangla strings), '
    '"duration" (string, e.g. "৩ দিন" or ""), '
    '"severity" (one of: mild, moderate, severe, unknown), '
    '"tests_done" (array of strings the patient mentioned having done), '
    '"possible_conditions" (array of likely conditions as English medical names, '
    "most likely first), "
    '"summary" (one or two English sentences summarizing the case for the doctor). '
    "Use an empty string or empty array when something is unknown. "
    "Base everything strictly on what the patient said; do not invent facts or "
    "give treatment advice."
)


def extract_case(history: list[dict]) -> dict:
    """Turn a {"user","assistant"} conversation into a structured case dict."""
    if not history:
        return {}
    convo = "\n".join(
        f"রোগী: {t['user']}\nসহকারী: {t['assistant']}" for t in history
    )
    client = _groq_client()  # Groq is reliable/free and supports JSON mode
    resp = client.chat.completions.create(
        model=GROQ_MODEL_ID,
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": convo},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=600,
    )
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
