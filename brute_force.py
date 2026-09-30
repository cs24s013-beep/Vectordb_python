"""
Brute Force search
===================
The simplest possible way to find the k nearest neighbours of a query
vector: measure the distance to *every* stored vector, sort, keep the
smallest k.

Complexity:  O(N * d)   for N stored vectors of dimension d, per query.
It is always exactly correct (100% recall) because it never skips
anything — every other algorithm in this project is really just a
clever way to *avoid* doing this full scan while still getting
(approximately) the same answer.

This is also the baseline the /benchmark endpoint times against, so you
can see with your own eyes how much HNSW/KD-Tree save you as the
dataset grows.
"""

from typing import Callable, List, Tuple
import numpy as np

from .models import VectorItem

DistFn = Callable[[np.ndarray, np.ndarray], float]


class BruteForce:
    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, item: VectorItem) -> None:
        self.items.append(item)

    def knn(self, q: np.ndarray, k: int, dist_fn: DistFn) -> List[Tuple[float, int]]:
        """Return the k closest (distance, id) pairs, sorted ascending."""
        scored = [(dist_fn(q, item.emb), item.id) for item in self.items]
        scored.sort(key=lambda pair: pair[0])
        return scored[:k]

    def remove(self, item_id: int) -> None:
        self.items = [item for item in self.items if item.id != item_id]
