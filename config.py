from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    llm_provider: Literal["ollama", "openai", "anthropic", "gemini"] = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "aya-expanse:8b"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    stt_provider: Literal["openai", "groq", "mlx", "gemini"] = "openai"
    groq_api_key: str = ""
    mlx_whisper_model: str = "base"

    tts_provider: Literal["edge-tts", "gtts"] = "edge-tts"
    tts_voice: str = "bn-BD-NabanitaNeural"

    port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
