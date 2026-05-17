"""LLM abstraction: Ollama (local) | OpenAI | Anthropic — switchable via config."""
import re
import httpx
from config import settings
from core.medical import SYSTEM_PROMPT


def _clean_response(text: str) -> str:
    """Strip markdown formatting and detect/truncate runaway repetition."""
    # Detect repetition: if any 4+ char substring repeats 10+ times consecutively
    repetition = re.search(r'(.{4,}?)\1{10,}', text)
    if repetition:
        text = text[:repetition.start()].strip()

    # Strip markdown bold/italic
    text = re.sub(r'\*+([^*]*)\*+', r'\1', text)
    # Strip markdown headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Strip numbered list markers at line start (keep the text)
    text = re.sub(r'^\d+\.\s+\*\*', '', text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


async def call_llm(messages: list[dict], system: str = SYSTEM_PROMPT) -> str:
    provider = settings.llm_provider

    if provider == "anthropic":
        return await _call_anthropic(messages, system)
    elif provider == "openai":
        return await _call_openai(messages, system)
    elif provider == "gemini":
        return await _call_gemini(messages, system)
    else:
        return await _call_ollama(messages, system)


async def _call_ollama(messages: list[dict], system: str) -> str:
    payload = {
        "model": settings.ollama_model,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 400,
            "repeat_penalty": 1.4,
            "repeat_last_n": 64,
        },
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{settings.ollama_url}/api/chat",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return _clean_response(data["message"]["content"])


async def _call_openai(messages: list[dict], system: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}] + messages,
        temperature=0.7,
        max_tokens=512,
    )
    return _clean_response(resp.choices[0].message.content)


async def _call_anthropic(messages: list[dict], system: str) -> str:
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    resp = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=system,
        messages=messages,
    )
    return _clean_response(resp.content[0].text)


async def _call_gemini(messages: list[dict], system: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)

    history = []
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        history.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=400,
        temperature=0.7,
    )
    chat = client.aio.chats.create(model=settings.gemini_model, history=history, config=config)
    resp = await chat.send_message(messages[-1]["content"])
    return _clean_response(resp.text)


async def get_llm_response(session: dict, user_input: str) -> str:
    """Add user turn, call LLM, return assistant reply."""
    messages = session.get("messages", [])
    messages.append({"role": "user", "content": user_input})

    response = await call_llm(messages)

    messages.append({"role": "assistant", "content": response})
    session["messages"] = messages

    from core.medical import extract_summary, strip_summary_block

    summary = extract_summary(response)
    # Always strip the raw summary block — even if extraction failed (truncated response)
    display_text = strip_summary_block(response)

    if summary:
        session["summary"] = summary
        session["stage"] = "complete"
        return display_text or "আপনার তথ্য সংগ্রহ সম্পন্ন হয়েছে। ডাক্তার শীঘ্রই আপনাকে ডাকবেন।"

    # Advance stage after first real exchange
    if session["stage"] == "intro":
        session["stage"] = "followup"

    return display_text
