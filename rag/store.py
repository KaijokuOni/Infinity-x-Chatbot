"""ChromaDB vector store for medical knowledge retrieval — cloud chatbot version."""
from __future__ import annotations
from pathlib import Path

from rag.knowledge import KNOWLEDGE_BASE

try:
    from app.utils.logger import get_logger  # type: ignore
    log = get_logger(__name__)
except Exception:
    import logging
    log = logging.getLogger(__name__)

_DB_DIR = Path(__file__).parent.parent / "data" / "rag_db"
_COLLECTION_NAME = "medical_knowledge"
_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_TOP_K = 2

_collection = None


def _build_document(entry: dict) -> str:
    parts = [
        f"রোগ: {entry['condition_bn']} ({entry['condition_en']})",
        f"লক্ষণ: {', '.join(entry['symptoms_bn'])}",
        f"ট্রিগার: {entry['trigger']}",
    ]
    if entry.get("red_flags"):
        parts.append(f"বিপদ চিহ্ন: {', '.join(entry['red_flags'])}")
    if entry.get("context"):
        parts.append(f"প্রেক্ষাপট: {entry['context']}")
    return "\n".join(parts)


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection

    import chromadb
    from chromadb.utils import embedding_functions

    _DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(_DB_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=_EMBED_MODEL)

    existing = [c.name for c in client.list_collections()]
    if _COLLECTION_NAME in existing:
        _collection = client.get_collection(_COLLECTION_NAME, embedding_function=ef)
        log.info(f"RAG: loaded ({_collection.count()} docs)")
    else:
        log.info("RAG: building vector store (first run ~30s)…")
        _collection = client.create_collection(_COLLECTION_NAME, embedding_function=ef)
        docs, ids, metas = [], [], []
        for entry in KNOWLEDGE_BASE:
            docs.append(_build_document(entry))
            ids.append(entry["id"])
            metas.append({
                "condition_bn": entry["condition_bn"],
                "condition_en": entry["condition_en"],
                "severity": entry["severity"],
                "tests": " | ".join(entry.get("tests", [])),
                "followup": " | ".join(entry.get("followup_questions", [])),
                "red_flags": " | ".join(entry.get("red_flags", [])),
            })
        _collection.add(documents=docs, ids=ids, metadatas=metas)
        log.info(f"RAG: indexed {len(docs)} entries")

    return _collection


def build_context_block(user_text: str, history: list[dict]) -> str:
    """Return a compact RAG context string to inject into the LLM system prompt."""
    if not history:
        return ""
    try:
        recent = " ".join(t["user"] for t in history[-3:])
        query = f"{recent} {user_text}".strip()
        col = _get_collection()
        results = col.query(query_texts=[query], n_results=min(_TOP_K, col.count()))
        ids = results["ids"][0]
        distances = results["distances"][0]

        lines = ["[চিকিৎসা তথ্য]"]
        for eid, dist in zip(ids, distances):
            if dist >= 1.5:
                continue
            entry = next((e for e in KNOWLEDGE_BASE if e["id"] == eid), None)
            if not entry:
                continue
            lines.append(f"রোগ: {entry['condition_bn']} (তীব্রতা: {entry['severity']})")
            if entry.get("red_flags"):
                lines.append(f"বিপদ: {', '.join(entry['red_flags'][:2])}")
            if entry.get("followup_questions"):
                lines.append(f"প্রশ্ন: {' | '.join(entry['followup_questions'][:2])}")
            if entry.get("tests"):
                lines.append(f"পরীক্ষা: {', '.join(entry['tests'][:3])}")
        lines.append("[এই তথ্য ব্যবহার করে প্রশ্ন করো]")
        return "\n".join(lines) if len(lines) > 2 else ""
    except Exception as exc:
        log.warning(f"RAG failed: {exc}")
        return ""
