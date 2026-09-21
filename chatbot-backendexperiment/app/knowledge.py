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
from pathlib import Path
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
    program: Optional[str] = None
    grade_range: Optional[str] = None
    subjects: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)

    @property
    def searchable_text(self) -> str:
        """All text used to build the retrieval index for this entry."""
        parts = [self.question, self.answer, *self.phrasings, *self.keywords, *self.items, *self.subjects, *self.features]
        if self.grade_range:
            parts.append(self.grade_range)
        return " ".join(parts)


class KnowledgeBaseError(RuntimeError):
    """Raised when the knowledge base file is missing or malformed."""


def load_entries() -> List[KBEntry]:
    target_path = Path(config.KNOWLEDGE_FILE) if config.KNOWLEDGE_FILE else None
    raw = None
    last_exc = None

    if target_path and target_path.exists():
        try:
            raw = target_path.read_text(encoding="utf-8-sig")
        except Exception as e:
            raise KnowledgeBaseError(f"Could not read knowledge base file from {target_path}: {e}") from e
    elif target_path:
        default_path = (config.PROJECT_ROOT / "knowledge" / "wementors_kb.json").resolve()
        try:
            is_custom = target_path.resolve() != default_path
        except Exception:
            is_custom = True

        if is_custom:
            raise KnowledgeBaseError(f"Could not read knowledge base file from {target_path}: file does not exist")

        fallbacks = [
            Path(__file__).resolve().parent / "wementors_kb.json",
            Path(__file__).resolve().parents[2] / "knowledge" / "wementors_kb.json",
            Path("api/wementors_kb.json"),
            Path("knowledge/wementors_kb.json"),
        ]
        for cand in fallbacks:
            if cand and cand.exists():
                try:
                    raw = cand.read_text(encoding="utf-8-sig")
                    break
                except Exception as e:
                    last_exc = e
        if raw is None:
            raise KnowledgeBaseError(f"Could not read knowledge base file from any candidate path: {last_exc or 'not found'}")
    else:
        raise KnowledgeBaseError("No knowledge base path configured")

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
                    program=item.get("program"),
                    grade_range=item.get("grade_range"),
                    subjects=item.get("subjects", []) or [],
                    features=item.get("features", []) or [],
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
