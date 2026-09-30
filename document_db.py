"""
DocumentDB
===========
Same idea as VectorDB, but for real documents embedded by Ollama
(768 dimensions for nomic-embed-text, versus the 16D toy vectors in
VectorDB) and used to power the RAG (Retrieval-Augmented Generation)
pipeline in Tab 3 of the UI.

Only HNSW + BruteForce are used here (no KD-Tree) because at 768
dimensions KD-Tree's pruning basically never triggers (see kdtree.py's
docstring) — it would just add tree-traversal overhead on top of what
is effectively still a full scan. BruteForce is kept as a fallback for
very small document sets (<10 chunks) since HNSW's graph structure
doesn't have enough data yet to meaningfully speed anything up at
that scale, and small-N brute force is already instant.

The distance metric is hard-coded to cosine (not user-selectable, unlike
the demo tab) because cosine is the standard, well-understood metric
for semantic text embeddings.
"""

import threading
from typing import List, Tuple
import numpy as np

from .models import VectorItem
from .distances import cosine
from .brute_force import BruteForce
from .hnsw import HNSW


class DocItem:
    __slots__ = ("id", "title", "text", "emb")

    def __init__(self, id_: int, title: str, text: str, emb: np.ndarray):
        self.id = id_
        self.title = title
        self.text = text
        self.emb = emb


class DocumentDB:
    def __init__(self):
        self.store = {}
        self.hnsw = HNSW(m=16, ef_construction=200)
        self.bf = BruteForce()
        self._lock = threading.Lock()
        self._next_id = 1
        self.dims = 0  # set from the first embedding Ollama returns

    def insert(self, title: str, text: str, emb: np.ndarray) -> int:
        with self._lock:
            if self.dims == 0:
                self.dims = emb.shape[0]
            item = DocItem(self._next_id, title, text, emb)
            self._next_id += 1
            self.store[item.id] = item

            vec_item = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(vec_item, cosine)
            self.bf.insert(vec_item)
            return item.id

    def search(self, q: np.ndarray, k: int, max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        with self._lock:
            if not self.store:
                return []
            if len(self.store) < 10:
                raw = self.bf.knn(q, k, cosine)
            else:
                raw = self.hnsw.knn(q, k, ef=50, dist_fn=cosine)

            out = []
            for dist, item_id in raw:
                doc = self.store.get(item_id)
                # max_dist filters out chunks that are only weakly
                # related — better to answer with "no relevant context"
                # than to feed the LLM something irrelevant.
                if doc is not None and dist <= max_dist:
                    out.append((dist, doc))
            return out

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.hnsw.remove(item_id)
            self.bf.remove(item_id)
            return True

    def all(self) -> List[DocItem]:
        with self._lock:
            return list(self.store.values())

    def size(self) -> int:
        with self._lock:
            return len(self.store)
