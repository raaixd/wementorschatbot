"""
Knowledge base loading.

The knowledge base lives in knowledge/wementors_kb.json as a flat list of
entries (see the "_meta.schema" key in that file for the format). This
module is the only place that knows how to read it, so the storage format
could change later (e.g. to a real CMS or database table) without touching
retrieval or conversation logic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from . import config

logger = logging.getLogger("wementors.knowledge")


@dataclass
class KBEntry:
    id: str
    category: str
    question: str
    answer: str
    phrasings: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    follow_ups: List[str] = field(default_factory=list)
    confidence: str = "verified"
    list_ids: List[str] = field(default_factory=list)
    """Ordered ids of related entries, e.g. the programs an overview entry
    enumerates. Used for reference resolution ("the first one", "the
    second program")."""
    format: str = "text"
    """How the answer should be presented: 'text' (plain sentence(s)),
    'bullets' (answer as an intro line + unordered items), or 'steps'
    (answer as an intro line + numbered items). See conversation.py's
    answer formatter."""
    items: List[str] = field(default_factory=list)
    """Bullet/step items used when format is 'bullets' or 'steps'."""

    @property
    def searchable_text(self) -> str:
        """All text used to build the retrieval index for this entry."""
        parts = [self.question, self.answer, *self.phrasings, *self.keywords, *self.items]
        return " ".join(parts)


class KnowledgeBaseError(RuntimeError):
    """Raised when the knowledge base file is missing or malformed."""


def load_entries() -> List[KBEntry]:
    try:
        raw = config.KNOWLEDGE_FILE.read_text(encoding="utf-8-sig")
    except (FileNotFoundError, OSError) as exc:
        raise KnowledgeBaseError(f"Could not read knowledge base file: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise KnowledgeBaseError(f"Knowledge base file is not valid JSON: {exc}") from exc

    raw_entries = data.get("entries", [])
    if not isinstance(raw_entries, list) or not raw_entries:
        raise KnowledgeBaseError("Knowledge base file has no entries")

    entries: List[KBEntry] = []
    seen_ids: set[str] = set()
    for item in raw_entries:
        try:
            entry_id = item["id"]
            if entry_id in seen_ids:
                logger.warning("Duplicate knowledge base id skipped: %s", entry_id)
                continue
            seen_ids.add(entry_id)
            entries.append(
                KBEntry(
                    id=entry_id,
                    category=item.get("category", "faq"),
                    question=item["question"],
                    answer=item["answer"],
                    phrasings=item.get("phrasings", []) or [],
                    keywords=item.get("keywords", []) or [],
                    follow_ups=item.get("follow_ups", []) or [],
                    confidence=item.get("confidence", "verified"),
                    list_ids=item.get("list_ids", []) or [],
                    format=item.get("format", "text"),
                    items=item.get("items", []) or [],
                )
            )
        except KeyError as exc:
            logger.warning("Skipping malformed knowledge base entry (missing %s): %r", exc, item)

    if not entries:
        raise KnowledgeBaseError("Knowledge base parsed but contained no usable entries")

    return entries


def get_entry_by_id(entries: List[KBEntry], entry_id: str) -> Optional[KBEntry]:
    for entry in entries:
        if entry.id == entry_id:
            return entry
    return None
