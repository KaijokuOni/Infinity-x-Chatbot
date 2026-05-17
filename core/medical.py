"""Medical system prompt and structured summary extraction."""

SYSTEM_PROMPT = """You are ভীণা, a warm and empathetic health assistant for OmniCare hospital.

CRITICAL RULES:
- Always respond in Bengali (বাংলা) using natural, simple spoken language.
- Never use markdown: no **, no ##, no bullet points, no numbered lists. Plain sentences only.
- Ask only ONE question per message. Wait for the answer before asking the next.
- Keep responses short — 2 to 3 sentences maximum.

CONVERSATION FLOW:
1. First ask the patient to describe their problem.
2. Then ask: how long have they had it?
3. Then ask: severity from 1 to 10?
4. Then ask: any existing conditions or medications?
5. Once you have all four answers, give a short farewell sentence, then output the summary block below.

SUMMARY FORMAT (output this only when you have collected all info):
[SUMMARY_START]
বিভাগ: <one of: শ্বাসতন্ত্র | হৃদরোগ | পরিপাকতন্ত্র | স্নায়ুতন্ত্র | হাড় ও জোড়া | চর্মরোগ | মূত্রতন্ত্র | জ্বর/সংক্রমণ | মানসিক স্বাস্থ্য | সাধারণ>
প্রধান সমস্যা: <chief complaint>
সময়কাল: <duration>
তীব্রতা: <1-10>
লক্ষণসমূহ: <comma separated>
পূর্ববর্তী ইতিহাস: <history or নেই>
ডাক্তারের জন্য নোট: <brief clinical note in Bengali>
[SUMMARY_END]
"""

INTRO_MESSAGE = (
    "আমি ভীণা, আপনার OmniCare স্বাস্থ্য সহকারী। "
    "আপনার সমস্যার কথা বিস্তারিত বলুন — আমি মনোযোগ দিয়ে শুনছি।"
)


def extract_summary(response_text: str) -> dict | None:
    """Parse [SUMMARY_START]...[SUMMARY_END] block from LLM response."""
    start = response_text.find("[SUMMARY_START]")
    end = response_text.find("[SUMMARY_END]")
    if start == -1 or end == -1:
        return None

    block = response_text[start + len("[SUMMARY_START]"):end].strip()
    result = {}
    key_map = {
        "বিভাগ": "category",
        "প্রধান সমস্যা": "chief_complaint",
        "সময়কাল": "duration",
        "তীব্রতা": "severity",
        "লক্ষণসমূহ": "symptoms",
        "পূর্ববর্তী ইতিহাস": "history",
        "ডাক্তারের জন্য নোট": "doctor_note",
    }

    for line in block.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if key in key_map:
                result[key_map[key]] = val

    return result if result else None


def strip_summary_block(response_text: str) -> str:
    """Remove the summary block from the response so only the conversational part remains."""
    start = response_text.find("[SUMMARY_START]")
    if start == -1:
        return response_text
    return response_text[:start].strip()


async def extract_patient_summary(session: dict) -> dict:
    """Generate or return cached summary for a session."""
    if session.get("summary"):
        return session["summary"]

    # If no summary was auto-generated, build one from conversation history
    from core.llm import call_llm

    history = session.get("messages", [])
    if not history:
        return {"error": "No conversation data"}

    prompt = (
        "নিচের কথোপকথনের উপর ভিত্তি করে একটি সংক্ষিপ্ত চিকিৎসা সারসংক্ষেপ তৈরি করুন।\n\n"
        + "\n".join(
            f"{'রোগী' if m['role'] == 'user' else 'ভীণা'}: {m['content']}"
            for m in history
        )
        + "\n\nঅনুগ্রহ করে [SUMMARY_START]...[SUMMARY_END] ফরম্যাটে সারসংক্ষেপ দিন।"
    )

    response = await call_llm([{"role": "user", "content": prompt}])
    summary = extract_summary(response)
    if summary:
        session["summary"] = summary
    return summary or {"error": "Could not extract summary"}
