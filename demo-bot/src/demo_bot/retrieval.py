"""Keyword search over knowledge base chunks, ranked with Okapi BM25."""

import heapq
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

_K1 = 1.5  # how quickly repeated terms stop adding to the score
_B = 0.75  # how strongly long chunks are penalised
_TOKEN = re.compile(r"[^\W_]+")  # letters and digits in any language
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
        term_counts = [Counter(tokenize(f"{c.title}\n{c.text}")) for c in self._chunks]
        lengths = [counts.total() for counts in term_counts]
        average_length = sum(lengths) / len(lengths) or 1.0
        self._length_norms = [1 - _B + _B * length / average_length for length in lengths]

        # An inverted index: for each term, the chunks holding it and how often.
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for position, counts in enumerate(term_counts):
            for term, frequency in counts.items():
                self._postings[term].append((position, frequency))

        # Lucene's variant of IDF, which stays positive for terms found in most chunks.
        chunk_count = len(self._chunks)
        self._idf = {
            term: math.log(1 + (chunk_count - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return up to `top_k` chunks that share at least one term with the query, best first."""
        scores: dict[int, float] = defaultdict(float)
        for term in sorted(set(tokenize(query))):
            for position, frequency in self._postings.get(term, ()):
                scores[position] += self._term_score(frequency, position, term)
        # Earlier chunks win ties, so results stay the same from one run to the next.
        best = heapq.nlargest(top_k, scores.items(), key=lambda entry: (entry[1], -entry[0]))
        return [SearchResult(self._chunks[position], score) for position, score in best]

    def _term_score(self, frequency: int, position: int, term: str) -> float:
        length_norm = self._length_norms[position]
        return self._idf[term] * frequency * (_K1 + 1) / (frequency + _K1 * length_norm)
