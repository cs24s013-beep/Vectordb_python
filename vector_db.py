"""
VectorDB
=========
The "front desk" for the 16-dimensional demo dataset. It doesn't
implement any search logic itself — it owns one BruteForce, one
KDTree, and one HNSW index, keeps them all in sync on every
insert/delete, and lets the caller pick which one answers a given
search (that's how the UI's "compare all algorithms" feature works:
same data, three engines, one call each).

A threading.Lock guards every method because Flask's dev server can
handle requests on multiple threads at once (we run it with
threaded=True) — without the lock, two inserts arriving at the same
instant could corrupt the shared dictionaries/trees. This plays the
same role as std::mutex in the original C++.
"""

import threading
import time
from typing import Callable, Dict, List
import numpy as np

from .models import VectorItem
from .distances import get_dist_fn
from .brute_force import BruteForce
from .kdtree import KDTree
from .hnsw import HNSW

DistFn = Callable[[np.ndarray, np.ndarray], float]


class VectorDB:
    def __init__(self, dims: int):
        self.dims = dims
        self.store: Dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(m=16, ef_construction=200)
        self._lock = threading.Lock()
        self._next_id = 1

    def insert(self, metadata: str, category: str, emb: np.ndarray, dist_fn: DistFn) -> int:
        with self._lock:
            item = VectorItem(self._next_id, metadata, category, emb)
            self._next_id += 1
            self.store[item.id] = item
            self.bf.insert(item)
            self.kdt.insert(item)
            self.hnsw.insert(item, dist_fn)
            return item.id

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.bf.remove(item_id)
            self.hnsw.remove(item_id)
            # KD-Tree can't delete in place cleanly -> rebuild from
            # what's left (see KDTree.rebuild's docstring for why).
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q: np.ndarray, k: int, metric: str, algo: str) -> dict:
        with self._lock:
            dist_fn = get_dist_fn(metric)
            t0 = time.perf_counter()

            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dist_fn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dist_fn)
            else:
                raw = self.hnsw.knn(q, k, ef=50, dist_fn=dist_fn)

            elapsed_us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for dist, item_id in raw:
                item = self.store.get(item_id)
                if item is not None:
                    hits.append(
                        {"id": item_id, "meta": item.metadata, "cat": item.category,
                         "emb": item.emb, "dist": dist}
                    )
            return {"hits": hits, "us": elapsed_us, "algo": algo, "metric": metric}

    def benchmark(self, q: np.ndarray, k: int, metric: str) -> dict:
        with self._lock:
            dist_fn = get_dist_fn(metric)

            def timed(fn) -> int:
                t0 = time.perf_counter()
                fn()
                return int((time.perf_counter() - t0) * 1_000_000)

            bf_us = timed(lambda: self.bf.knn(q, k, dist_fn))
            kd_us = timed(lambda: self.kdt.knn(q, k, dist_fn))
            hnsw_us = timed(lambda: self.hnsw.knn(q, k, 50, dist_fn))
            return {"bfUs": bf_us, "kdUs": kd_us, "hnswUs": hnsw_us, "n": len(self.store)}

    def all(self) -> List[VectorItem]:
        with self._lock:
            return list(self.store.values())

    def hnsw_info(self) -> dict:
        with self._lock:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self._lock:
            return len(self.store)
