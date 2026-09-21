"""
Retrieval layer for the RAG pipeline.

This implements a small hybrid retriever entirely in the standard library:

  1. Each knowledge-base entry is tokenized and turned into a TF-IDF
     weighted vector (question + phrasings + keywords + answer).
  2. The user's query (plus a little recent conversation context) is
     vectorized the same way.
  3. Cosine similarity ranks entries against the query.
  4. A small keyword-overlap bonus rewards entries whose question text
     shares literal words with the query, which helps short/ambiguous
     queries where TF-IDF alone is noisy.

This is intentionally *not* a neural embedding model — the knowledge base
is a few dozen short FAQ entries, so a lexical method is fast, has zero
extra dependencies, and is easy for a student to read and extend. If the
knowledge base grows much larger or needs true semantic matching (e.g.
paraphrases with no shared words), swap `_vectorize` for a call to an
embeddings API — the rest of the pipeline (scoring, thresholding,
top-k, dedup) does not need to change.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

from .knowledge import KBEntry

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "i", "you", "he", "she",
    "it", "we", "they", "what", "which", "who", "whom", "where", "when",
    "this", "that", "these", "those", "to", "of", "in", "on", "for", "and",
    "or", "but", "with", "about", "how", "can", "will", "would", "should",
    "could", "my", "your", "our", "me", "us", "at", "as", "by", "from",
    # Possessives/pronouns carry no retrieval signal but DO count against
    # the coverage gate in search() if left in, which would wrongly sink
    # legitimate follow-ups like "What are its fees?" (see the
    # reference-resolution tests).
    "its", "their", "his", "her", "them", "theirs", "ours", "yours",
    "there", "here", "any", "some", "please", "tell",
    # Query framing verbs that carry no domain content and would wrongly
    # penalize multi-facet question coverage:
    "find", "need", "want", "looking", "look", "get", "give", "provide", "assign",
}

_TARGET_STEMS = {
    "boards": "board",
    "exams": "exam",
    "prepping": "prepare",
    "preparation": "prepare",
    "math": "maths",
    "mathematics": "maths",
    "mentors": "mentor",
}


def _stem_token(t: str) -> str:
    return _TARGET_STEMS.get(t, t)


def tokenize(text: str) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    # Single characters are dropped as noise, EXCEPT digits: grade numbers
    # are single digits ("class 8", "grade 5") and are the most important
    # word in such a query. Dropping them made "class 8" retrieve the
    # free-demo-class entry instead of anything about grade 8.
    result = []
    for t in tokens:
        if t in _STOPWORDS:
            continue
        if len(t) <= 1 and not t.isdigit():
            continue
        stemmed = _stem_token(t)
        if stemmed not in _STOPWORDS:
            result.append(stemmed)
    return result


_FEE_TRIGGER_WORDS = {
    "fee", "fees", "cost", "costs", "price", "prices", "pricing",
    "charge", "charges", "rate", "rates", "tuition", "inr", "rupee",
    "rupees", "pay", "payment", "expensive", "affordable", "afford",
    "discount", "discounts", "scholarship", "scholarships",
}


@dataclass
class ScoredEntry:
    entry: KBEntry
    score: float


class Retriever:
    """Builds a TF-IDF index over a list of KBEntry objects at startup
    (or whenever the knowledge base is reloaded) and answers similarity
    queries against it."""

    def __init__(self, entries: List[KBEntry]):
        self.entries = entries
        self._doc_tokens: List[List[str]] = [tokenize(e.searchable_text) for e in entries]
        self._doc_token_sets: List[set] = [set(tokens) for tokens in self._doc_tokens]
        self._idf: Dict[str, float] = self._build_idf(self._doc_tokens)
        self._doc_vectors: List[Dict[str, float]] = [
            self._vectorize(tokens) for tokens in self._doc_tokens
        ]
        self._doc_norms: List[float] = [self._norm(vec) for vec in self._doc_vectors]
        # Precompute the plain token set of each question, for the keyword bonus.
        self._question_tokens: List[set] = [set(tokenize(e.question)) for e in entries]
        # Tokens from the entry's *intent* fields only (question, alternative
        # phrasings, keywords) — i.e. the words that signal "this entry is
        # about that topic", as opposed to incidental words that merely
        # happen to occur in the answer's prose. Used by the intent-field
        # check in search().
        self._intent_tokens: List[set] = [
            set(tokenize(" ".join([e.question, *e.phrasings, *e.keywords])))
            for e in entries
        ]

    def _build_idf(self, doc_tokens: List[List[str]]) -> Dict[str, float]:
        n_docs = max(len(doc_tokens), 1)
        doc_freq: Counter = Counter()
        for tokens in doc_tokens:
            for term in set(tokens):
                doc_freq[term] += 1
        return {
            term: math.log((1 + n_docs) / (1 + freq)) + 1.0
            for term, freq in doc_freq.items()
        }

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        return {
            term: (count / total) * self._idf.get(term, math.log(2))
            for term, count in counts.items()
        }

    @staticmethod
    def _norm(vector: Dict[str, float]) -> float:
        return math.sqrt(sum(v * v for v in vector.values())) or 1e-9

    def _cosine(self, query_vec: Dict[str, float], query_norm: float, index: int) -> float:
        doc_vec = self._doc_vectors[index]
        if len(query_vec) < len(doc_vec):
            shared = query_vec
            other = doc_vec
        else:
            shared = doc_vec
            other = query_vec
        dot = sum(weight * other.get(term, 0.0) for term, weight in shared.items())
        denom = query_norm * self._doc_norms[index]
        return dot / denom if denom else 0.0

    def search(self, query: str, top_k: int = 3) -> List[ScoredEntry]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_vec = self._vectorize(query_tokens)
        query_norm = self._norm(query_vec)
        query_token_set = set(query_tokens)
        lowered_query = query.lower()
        has_fee_word = bool(query_token_set & _FEE_TRIGGER_WORDS or "how much" in lowered_query)

        scored: List[ScoredEntry] = []
        for index, entry in enumerate(self.entries):
            # Category gate: fee entries must never match queries that do not
            # explicitly ask about fees, costs, or pricing.
            if (entry.category == "fees" or entry.id == "fees-and-pricing") and not has_fee_word:
                continue

            # Middle school gate: generic 'school' in non-school context must not match Middle School
            if (entry.id.startswith("middle-school-") or entry.id == "program-middle-school"):
                if not any(k in lowered_query for k in ("middle", "middel", "midle", "class 6", "class 7", "class 8", "grade 6", "grade 7", "grade 8")):
                    if any(k in lowered_query for k in ("not in school", "without being in school", "without school", "outside of school")):
                        continue

            # Demo booking gate: how-to-book-demo must not match on 'join' or 'where' alone without booking intent
            if entry.id == "how-to-book-demo":
                if not any(k in lowered_query for k in ("book", "demo", "trial", "register", "schedule", "sign up", "signup", "join")):
                    continue

            # Confident speaker gate: queries explicitly asking about academic courses/subjects/grades
            # must not match Confident Speaker entries unless Confident Speaker or speaking is also mentioned
            if entry.id.startswith("confident-speaker-") or entry.id == "program-confident-speaker":
                has_academic_word = any(
                    w in lowered_query
                    for w in (
                        "academic", "academics", "school student", "school course",
                        "maths", "mathematics", "science", "social studies",
                        "middle school", "foundation years", "board exam",
                        "grade 3", "grade 4", "grade 5", "grade 6", "grade 7", "grade 8", "grade 9", "grade 10",
                        "class 3", "class 4", "class 5", "class 6", "class 7", "class 8", "class 9", "class 10",
                    )
                )
                has_speaker_word = any(
                    w in lowered_query
                    for w in ("confident", "speaker", "speaking", "spoken", "speech", "interview")
                )
                if has_academic_word and not has_speaker_word:
                    continue

            # Grade 1-2 availability gate: must not match unless explicitly asking about 1st/2nd grade
            if entry.id == "grade-1-and-2-availability":
                has_g1_g2 = bool(
                    re.search(
                        r"\b((?:first|1st|second|2nd)\s+grades?|grades?\s*(?:1|2|one|two)\b(?!\s*[0-9])|"
                        r"(?:first|1st|second|2nd)\s+class(?:es)?|class\s*(?:1|2|one|two)\b(?!\s*[0-9])|"
                        r"(?:first|1st|second|2nd)\s+standards?|standards?\s*(?:1|2|one|two)\b(?!\s*[0-9])|"
                        r"grades?\s*(?:1\s*(?:and|&|or|to|-|–)\s*2|1\s*,\s*2)|"
                        r"classes\s*(?:1\s*(?:and|&|or|to|-|–)\s*2|1\s*,\s*2)|"
                        r"(?:first|1st)\s*(?:and|&|or|to|-|–)\s*(?:second|2nd)\s+grades?|"
                        r"(?:first|1st)\s+or\s+(?:second|2nd)\s+grade|(?:first|second)\s+grader)\b",
                        lowered_query,
                    )
                )
                if not has_g1_g2:
                    continue

            cosine = self._cosine(query_vec, query_norm, index)
            overlap = len(query_token_set & self._question_tokens[index])
            overlap_bonus = overlap * 0.08
            score = cosine + overlap_bonus

            # Phrasing / question match bonus: if query directly matches
            # an intended phrasing or question, grant a decisive boost.
            norm_q = lowered_query.strip("? .!").strip()
            has_ms_terms = any(k in lowered_query for k in ("middle", "class 6", "class 7", "class 8", "grade 6", "grade 7", "grade 8"))
            if any(norm_q == p.lower().strip("? .!").strip() for p in [entry.question, *entry.phrasings]):
                if entry.id == "middle-school-doubt-clinics" and not has_ms_terms:
                    pass
                else:
                    score += 0.5

            is_doubt_query = bool(re.search(r"\b(doubt\s+clinics?|doubt\s+solving|doubt[- ]clearing)\b", lowered_query))
            is_lab_query = bool(re.search(r"\b(practical\s+labs?|practical\s+problem\s+sets?)\b", lowered_query))

            # Grade-based program routing boosts (for program inquiries, not specific feature inquiries)
            if not is_doubt_query and not is_lab_query:
                if entry.id == "program-foundation-years":
                    if re.search(r"\b(grades?\s*[345]\b|class\s*[345]\b|[345](?:th|rd|st)?\s*(?:grade|class|standard)\b|foundation(?:\s+years?)?)\b", lowered_query):
                        score += 0.5
                elif entry.id == "program-middle-school":
                    if re.search(r"\b(grades?\s*[678]\b|class\s*[678]\b|[678]th\s*(?:grade|class|standard)\b|middle(?:\s+school)?)\b", lowered_query):
                        score += 0.5
                elif entry.id == "program-senior-school":
                    if re.search(r"\b(grades?\s*(?:9|10)\b|class\s*(?:9|10)\b|(?:9|10)th\s*(?:grade|class|standard)\b|senior(?:\s+school(?:\s+focus)?)?)\b", lowered_query):
                        score += 0.5
                elif entry.id == "program-confident-speaker":
                    if re.search(r"\b(confident\s+speaker|confident\s+speaking|spoken\s+english|public\s+speaking|interview\s+skills?)\b", lowered_query):
                        score += 0.5

            # General vs Middle School feature routing
            if is_doubt_query:
                if not has_ms_terms:
                    if entry.id == "doubt-solving-sessions":
                        score += 0.6
                    elif entry.id == "middle-school-doubt-clinics":
                        score *= 0.5
                else:
                    if entry.id == "middle-school-doubt-clinics":
                        score += 0.6

            if is_lab_query:
                if has_ms_terms and entry.id == "middle-school-practical-labs":
                    score += 0.6

            # Coverage gate: a single rare shared word (high IDF weight)
            # can otherwise make an unrelated multi-word query look like a
            # confident match (e.g. "financial aid" sharing only "offer"
            # with a programs entry, or "duration of the course" sharing
            # only "duration" with the demo-class-length entry). Scale the
            # score by what fraction of the query's words actually appear
            # anywhere in the entry, so genuinely unsupported questions
            # fall through to the honest fallback instead of returning a
            # plausible-looking but irrelevant answer. Applies to queries
            # of 2+ words; genuine matches score full coverage and are
            # unaffected.
            matched_tokens = query_token_set & self._doc_token_sets[index]
            if len(query_token_set) >= 2:
                coverage = len(matched_tokens) / len(query_token_set)
                score *= coverage

            # Intent-field check: an entry's answer prose contains
            # incidental words that don't describe what the entry is
            # *about* (e.g. the fees entry's answer happens to contain
            # "help"). Matching only on those is weak evidence, so halve
            # the score when none of the matched words appear in the
            # entry's question, alternative phrasings, or keywords. This
            # is what stops a vague "Can you help me?" from being answered
            # as a fees question.
            if matched_tokens and not (matched_tokens & self._intent_tokens[index]):
                score *= 0.5

            if score > 0:
                scored.append(ScoredEntry(entry=entry, score=round(score, 4)))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
