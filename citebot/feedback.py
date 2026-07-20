"""Append-only feedback log: one JSON line per 👍/👎 a user gives an answer.

This is the piece that closes the RAG loop (retrieval → generation →
evaluation → improvement from real usage): the fixed eval dataset
(citebot/evaluate.py) catches regressions, but only real feedback tells you
which live answers actually satisfied a user.
"""

import json
from datetime import datetime, timezone

from citebot.config import FEEDBACK_LOG_PATH


def record_feedback(question, standalone_question, answer, sources, usage, rating):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "standalone_question": standalone_question,
        "answer": answer,
        "sources": [
            {"file": s["file"], "flagged": s.get("flagged", False)} for s in sources
        ],
        "tokens": usage.get("total_tokens"),
        "cost_usd": usage.get("cost_usd"),
        "latency_seconds": usage.get("latency_seconds"),
        "rating": rating,
    }

    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
