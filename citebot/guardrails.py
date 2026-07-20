"""Heuristic prompt-injection detection for ingested document chunks.

Documents can come from third-party uploads, so a chunk might contain text
trying to hijack the assistant (e.g. "ignore previous instructions"). This is
a heuristic tagger, not a hard filter: a false positive should never silently
hide legitimate content, so flagged chunks are still indexed and retrievable
— just marked, so the prompt and the UI can treat them as untrusted data.
"""

import re

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the |prior |previous |above )+instructions",
    r"disregard (the |all |any )?(above|previous|prior) instructions",
    r"forget (all |your )?(previous |prior )?instructions",
    r"you are now (in )?(developer|admin|god) mode",
    r"reveal (your |the )?system prompt",
    r"print (your |the )?(system )?prompt",
    r"new instructions?:",
    r"system override",
    r"act as (if )?you (are|were)",
    r"do not (follow|obey) (the|your) (system|previous) instructions",
]

_COMPILED = [re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS]


def scan(text):
    """Return the first injection pattern matched in `text`, or None."""
    for pattern in _COMPILED:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def annotate_injection_risk(chunks):
    """Tag each chunk's metadata in place with whether it looks like a
    prompt-injection attempt, so the risk travels with the chunk through
    retrieval and can be surfaced in citations."""
    for chunk in chunks:
        match = scan(chunk.page_content)
        chunk.metadata["injection_flagged"] = bool(match)
        if match:
            chunk.metadata["injection_match"] = match
    return chunks
