"""Retrievers for few-shot example selection.

Both retrievers expose:
    retrieve(query: str, k: int, memory: List[Dict]) -> List[int]
        Returns indices into `memory`, sorted by similarity (most similar first).

MiniLMRetriever maintains an internal {question_text: embedding} cache that
must be kept in sync with the memory list — call refresh(memory) after appending
new records so the new question gets embedded before the next retrieve().
"""

from typing import Dict, List, Optional
import numpy as np
import Levenshtein

_MINI_LM_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class LevenshteinRetriever:
    def __init__(self):
        pass

    def refresh(self, memory: List[Dict]):
        pass

    def retrieve(self, query: str, k: int, memory: List[Dict]) -> List[int]:
        dists = {i: Levenshtein.distance(query, m["question"]) for i, m in enumerate(memory)}
        sorted_items = sorted(dists.items(), key=lambda x: x[1])
        return [idx for idx, _ in sorted_items[:k]]


class MiniLMRetriever:
    def __init__(self, model_name: str = _MINI_LM_NAME):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._cache: Dict[str, np.ndarray] = {}

    def _embed_one(self, text: str) -> np.ndarray:
        v = self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(v, dtype=np.float32)

    def refresh(self, memory: List[Dict]):
        new_texts = [m["question"] for m in memory if m["question"] not in self._cache]
        if not new_texts:
            return
        seen = set()
        unique_texts = []
        for t in new_texts:
            if t not in seen:
                seen.add(t)
                unique_texts.append(t)
        vecs = self._model.encode(unique_texts, normalize_embeddings=True)
        for t, v in zip(unique_texts, vecs):
            self._cache[t] = np.asarray(v, dtype=np.float32)

    def retrieve(self, query: str, k: int, memory: List[Dict]) -> List[int]:
        self.refresh(memory)
        if query not in self._cache:
            self._cache[query] = self._embed_one(query)
        q_vec = self._cache[query]
        sims = []
        for i, m in enumerate(memory):
            doc_vec = self._cache[m["question"]]
            sims.append((i, float(np.dot(q_vec, doc_vec))))
        sims.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sims[:k]]
