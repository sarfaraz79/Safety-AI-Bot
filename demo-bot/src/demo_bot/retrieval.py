"""Keyword search over knowledge base chunks, ranked with Okapi BM25."""

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

_K1 = 1.5  # how quickly repeated terms stop adding to the score
_B = 0.75  # how strongly long chunks are penalised
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from get has have how i if in is it me my "
    "of on or s so t than that the their this to was we what when where which who will with "
    "you your".split()
)


@dataclass(frozen=True, slots=True)
class Chunk:
    """One searchable section of a knowledge base document."""

    id: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    """Lower-case words and numbers, without common English stopwords."""
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS]


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise ValueError("The index needs at least one chunk.")
        self._chunks = tuple(chunks)
        self._term_counts = [Counter(tokenize(f"{c.title}\n{c.text}")) for c in self._chunks]
        self._lengths = [counts.total() for counts in self._term_counts]
        self._average_length = sum(self._lengths) / len(self._lengths) or 1.0

        chunk_count = len(self._chunks)
        document_frequency = Counter(term for counts in self._term_counts for term in counts)
        # Lucene's variant of IDF, which stays positive for terms found in most chunks.
        self._idf = {
            term: math.log(1 + (chunk_count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return up to `top_k` chunks that share at least one term with the query, best first."""
        terms = set(tokenize(query))
        results = []
        for chunk, counts, length in zip(
            self._chunks, self._term_counts, self._lengths, strict=True
        ):
            score = sum(self._term_score(counts[t], length, t) for t in terms if t in counts)
            if score > 0:
                results.append(SearchResult(chunk, score))
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def _term_score(self, frequency: int, length: int, term: str) -> float:
        length_norm = 1 - _B + _B * length / self._average_length
        return self._idf[term] * frequency * (_K1 + 1) / (frequency + _K1 * length_norm)
