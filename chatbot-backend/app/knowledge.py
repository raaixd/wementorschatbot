from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .config import settings


@dataclass
class KnowledgeEntry:
    id: str
    category: str
    question: str
    answer: str
    aliases: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    follow_ups: List[str] = field(default_factory=list)
    confidence: str = "verified"
    source: str = ""
    program_index: Optional[int] = None
    related_program_ids: List[str] = field(default_factory=list)

    def searchable_text(self) -> str:
        parts = [
            self.question,
            " ".join(self.aliases),
            " ".join(self.keywords),
            self.answer,
            self.category,
            self.id.replace("_", " ").replace("-", " "),
        ]
        return " ".join(parts)


@dataclass
class KnowledgeBase:
    entries: List[KnowledgeEntry]
    programs_order: List[str]
    restrictions: List[str]

    def by_id(self, entry_id: str) -> Optional[KnowledgeEntry]:
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None

    def by_category(self, category: str) -> List[KnowledgeEntry]:
        return [entry for entry in self.entries if entry.category == category]


_cached: Optional[KnowledgeBase] = None


def load_knowledge(path: Optional[Path] = None) -> KnowledgeBase:
    global _cached
    kb_path = path or settings.knowledge_path
    if _cached is not None and path is None:
        return _cached

    data = json.loads(kb_path.read_text(encoding="utf-8"))
    entries = []
    for item in data.get("entries", []):
        entries.append(
            KnowledgeEntry(
                id=item["id"],
                category=item["category"],
                question=item["question"],
                answer=item["answer"],
                aliases=item.get("aliases", []),
                keywords=item.get("keywords", []),
                follow_ups=item.get("follow_ups", []),
                confidence=item.get("confidence", "verified"),
                source=item.get("source", ""),
                program_index=item.get("program_index"),
                related_program_ids=item.get("related_program_ids", []),
            )
        )
    kb = KnowledgeBase(
        entries=entries,
        programs_order=data.get("programs_order", []),
        restrictions=data.get("restrictions", []),
    )
    if path is None:
        _cached = kb
    return kb


def reload_knowledge() -> KnowledgeBase:
    global _cached
    _cached = None
    return load_knowledge()
