from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Sequence

from .knowledge import KnowledgeBase, KnowledgeEntry


STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "is", "are",
    "do", "you", "your", "me", "my", "we", "i", "please", "tell", "about",
    "what", "which", "how", "can", "could", "would", "with", "from",
}

SYNONYMS = {
    "classes": ["grades", "programs"],
    "class": ["grade", "program"],
    "cost": ["fees", "price"],
    "price": ["fees", "cost"],
    "enroll": ["demo", "admission", "book"],
    "enrol": ["demo", "admission", "book"],
    "apply": ["demo", "admission", "book"],
    "join": ["demo", "admission"],
    "maths": ["mathematics", "math"],
    "math": ["mathematics", "maths"],
    "teacher": ["mentor", "teaching"],
    "course": ["program"],
    "courses": ["programs"],
    "timing": ["schedule", "hours"],
    "timings": ["schedule", "hours"],
}


@dataclass
class ScoredEntry:
    entry: KnowledgeEntry
    score: float


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    tokens: List[str] = []
    for word in words:
        if word in STOPWORDS or len(word) < 2:
            continue
        tokens.append(word)
        tokens.extend(SYNONYMS.get(word, []))
    return tokens


def _idf(df: int, n_docs: int) -> float:
    return math.log((1 + n_docs) / (1 + df)) + 1.0


def retrieve(
    query: str,
    kb: KnowledgeBase,
    top_k: int = 4,
    min_score: float = 4.0,
    category_hint: str = "",
) -> List[ScoredEntry]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    n_docs = max(len(kb.entries), 1)
    df = {}
    tokenized_docs = []
    for entry in kb.entries:
        tokens = tokenize(entry.searchable_text())
        tokenized_docs.append(tokens)
        unique = set(tokens)
        for token in unique:
            df[token] = df.get(token, 0) + 1

    scored: List[ScoredEntry] = []
    for entry, doc_tokens in zip(kb.entries, tokenized_docs):
        haystack = entry.searchable_text().lower()
        score = 0.0
        for token in query_tokens:
            tf = doc_tokens.count(token)
            if tf:
                score += (1.0 + math.log(tf)) * _idf(df.get(token, 0), n_docs)

        question = entry.question.lower()
        if question and question in query.lower():
            score += 8.0
        for alias in entry.aliases:
            if alias.lower() in query.lower():
                score += 6.0
        for keyword in entry.keywords:
            if keyword.lower() in query.lower():
                score += 2.5
        if category_hint and entry.category == category_hint:
            score += 3.0
        if all(token in haystack for token in query_tokens[:4]):
            score += 1.5
        scored.append(ScoredEntry(entry=entry, score=score))

    scored.sort(key=lambda item: item.score, reverse=True)
    filtered = [item for item in scored if item.score >= min_score]
    if not filtered:
        return []

    unique: List[ScoredEntry] = []
    seen = set()
    for item in filtered:
        if item.entry.id in seen:
            continue
        seen.add(item.entry.id)
        unique.append(item)
        if len(unique) >= top_k:
            break

    if unique:
        best = unique[0].score
        unique = [item for item in unique if item.score >= best * 0.45 or item.entry.category == unique[0].entry.category]
    return unique[:top_k]


def entries_only(scored: Sequence[ScoredEntry]) -> List[KnowledgeEntry]:
    return [item.entry for item in scored]
