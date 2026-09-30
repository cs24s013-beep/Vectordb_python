"""
KD-Tree (K-Dimensional Tree)
=============================
A binary tree that recursively splits vector space in half along one
dimension at a time, cycling through dimensions as you go deeper
(dimension 0 at depth 0, dimension 1 at depth 1, ... wrapping back to
0 at depth = dims).

Build (insert):
    Walk down the tree like a binary search tree, but the "compare"
    axis changes every level. At depth d, compare the point's value
    on axis `d % dims` against the current node's value on that same
    axis; go left if smaller, right otherwise.

Search (k nearest neighbours):
    Walk down to the leaf that *would* contain the query point,
    recording candidates in a bounded max-heap of size k as you go
    (the heap always holds the best k found so far, worst on top).
    Then walk back UP the recursion, and at each node ask: "could the
    OTHER side of this split possibly contain something closer than my
    current worst candidate?" This is the "ball within hyperslab"
    check — if the query point's distance to the splitting plane is
    already farther than our worst candidate, that whole subtree is
    skipped (pruned) because nothing in it can possibly be closer.

This pruning is what makes KD-Tree faster than brute force in LOW
dimensions (each level of the tree throws away roughly half the
remaining candidates, similar to binary search). It breaks down in
HIGH dimensions (~50+) because in high-dimensional space almost every
point ends up close to *some* splitting boundary, so the "could the
other side be closer?" check almost always says yes — nothing gets
pruned and you're back to a full scan, just with tree-traversal
overhead on top. That's why this project uses KD-Tree for the 16D demo
vectors but HNSW (not KD-Tree) for the 768D Ollama document embeddings.
"""

import heapq
from typing import Callable, List, Optional, Tuple
import numpy as np

from .models import VectorItem

DistFn = Callable[[np.ndarray, np.ndarray], float]


class _KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem):
        self.item = item
        self.left: Optional["_KDNode"] = None
        self.right: Optional["_KDNode"] = None


class KDTree:
    def __init__(self, dims: int):
        self.dims = dims
        self.root: Optional[_KDNode] = None

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, item: VectorItem) -> None:
        self.root = self._insert(self.root, item, depth=0)

    def _insert(self, node: Optional[_KDNode], item: VectorItem, depth: int) -> _KDNode:
        if node is None:
            return _KDNode(item)
        axis = depth % self.dims
        if item.emb[axis] < node.item.emb[axis]:
            node.left = self._insert(node.left, item, depth + 1)
        else:
            node.right = self._insert(node.right, item, depth + 1)
        return node

    def rebuild(self, items: List[VectorItem]) -> None:
        """
        KD-Trees don't support clean deletion (removing a node from the
        middle of the tree can break the splitting invariants), so the
        simplest correct approach — and what the original C++ does too
        — is: throw the whole tree away and re-insert everything that's
        left. Fine for a demo-sized dataset; a production KD-Tree would
        use lazy/soft deletion instead.
        """
        self.root = None
        for item in items:
            self.insert(item)

    # ------------------------------------------------------------------
    # k-Nearest-Neighbour search
    # ------------------------------------------------------------------

    def knn(self, q: np.ndarray, k: int, dist_fn: DistFn) -> List[Tuple[float, int]]:
        # We keep a bounded max-heap of size k using Python's heapq
        # (which is a MIN-heap) by negating distances. So the item with
        # the *largest* real distance sits at heap[0] — i.e. it's always
        # the first one we'd evict if we find something better. This
        # mirrors the C++ std::priority_queue (a max-heap by default).
        heap: List[Tuple[float, int]] = []  # (-distance, id)
        self._knn(self.root, q, k, depth=0, dist_fn=dist_fn, heap=heap)
        result = [(-neg_d, item_id) for neg_d, item_id in heap]
        result.sort(key=lambda pair: pair[0])
        return result

    def _knn(
        self,
        node: Optional[_KDNode],
        q: np.ndarray,
        k: int,
        depth: int,
        dist_fn: DistFn,
        heap: List[Tuple[float, int]],
    ) -> None:
        if node is None:
            return

        worst_in_heap = -heap[0][0] if heap else float("inf")
        d_here = dist_fn(q, node.item.emb)
        if len(heap) < k or d_here < worst_in_heap:
            heapq.heappush(heap, (-d_here, node.item.id))
            if len(heap) > k:
                heapq.heappop(heap)

        axis = depth % self.dims
        diff = q[axis] - node.item.emb[axis]
        # Recurse into the side of the split the query point falls on
        # first — it's the more promising side.
        closer, farther = (node.left, node.right) if diff < 0 else (node.right, node.left)
        self._knn(closer, q, k, depth + 1, dist_fn, heap)

        # The pruning check: only bother with the far side if it could
        # possibly contain something closer than our current worst.
        worst_in_heap = -heap[0][0] if heap else float("inf")
        if len(heap) < k or abs(diff) < worst_in_heap:
            self._knn(farther, q, k, depth + 1, dist_fn, heap)
