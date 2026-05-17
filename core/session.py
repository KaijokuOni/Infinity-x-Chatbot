"""In-memory session store with conversation state."""
import time
from threading import Lock
from core.medical import INTRO_MESSAGE


class SessionManager:
    def __init__(self, ttl_seconds: int = 3600):
        self._sessions: dict[str, dict] = {}
        self._lock = Lock()
        self._ttl = ttl_seconds

    def get_or_create(self, session_id: str) -> dict:
        with self._lock:
            self._evict_expired()
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    "id": session_id,
                    "stage": "intro",
                    "messages": [],
                    "summary": None,
                    "created_at": time.time(),
                    "last_active": time.time(),
                }
            else:
                self._sessions[session_id]["last_active"] = time.time()
            return self._sessions[session_id]

    def get(self, session_id: str) -> dict | None:
        with self._lock:
            return self._sessions.get(session_id)

    def add_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        # Messages are already appended inside llm.get_llm_response;
        # this just updates last_active timestamp.
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["last_active"] = time.time()

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "id": s["id"],
                    "stage": s["stage"],
                    "turns": len(s["messages"]) // 2,
                    "has_summary": s["summary"] is not None,
                    "last_active": s["last_active"],
                }
                for s in self._sessions.values()
            ]

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s["last_active"] > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]
