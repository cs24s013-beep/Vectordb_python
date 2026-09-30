"""
HNSW — Hierarchical Navigable Small World
===========================================
This is the algorithm real production vector databases (Pinecone,
Weaviate, Chroma, Milvus, Qdrant) actually use. It trades a small,
tunable amount of accuracy (it's "approximate" nearest-neighbour, not
exact like BruteForce/KDTree) for search speed that barely slows down
as the dataset grows into the millions.

THE BIG IDEA — a multilayer graph, like a subway map
------------------------------------------------------
Every stored vector is a "node" in a graph. Nodes are connected to
their approximate nearest neighbours. But instead of one flat graph,
HNSW builds several *layers*:

    Layer 2 (very few nodes)     o---------------o     <- express lines,
    Layer 1 (some nodes)         o---o-------o----o        long jumps
    Layer 0 (every node)     o-o-o-o-o-o-o-o-o-o-o-o-o  <- local stops,
                                                            every node

Each node is randomly assigned a "max layer" it belongs to (higher
layers are exponentially rarer — most nodes only exist at layer 0).
This mirrors a real subway system: express trains (upper layers) skip
most stops so you cover a lot of ground fast, then you switch to the
local line (layer 0) to reach your exact destination.

SEARCH: start at the top layer's single entry point, greedily walk
toward the query (like "which express stop gets me closest?"), drop
down a layer once you can't improve, repeat until you're at layer 0,
then do a slightly wider local search to collect the final k results.

INSERT: same greedy descent to find where a new node belongs, then
connect it to its M nearest neighbours at each layer it participates
in (bidirectionally), pruning a neighbour's connection list back down
to M if adding this new edge made it too crowded.

Why it's fast: most of the "distance" (in graph hops, not vector
space) is covered by a handful of long-range jumps on the sparse upper
layers, so you never have to look at more than a small, mostly
constant number of nodes regardless of how many millions are stored.
Compare to BruteForce, which must look at literally everything.
"""

import math
import heapq
import random
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from .models import VectorItem

DistFn = Callable[[np.ndarray, np.ndarray], float]


class _HNSWNode:
    __slots__ = ("item", "max_layer", "neighbors")

    def __init__(self, item: VectorItem, max_layer: int):
        self.item = item
        self.max_layer = max_layer
        # neighbors[layer] = list of neighbour ids at that layer
        self.neighbors: List[List[int]] = [[] for _ in range(max_layer + 1)]


class HNSW:
    def __init__(self, m: int = 16, ef_construction: int = 200, seed: int = 42):
        self.M = m                      # max neighbours per node (layers > 0)
        self.M0 = 2 * m                 # max neighbours per node at layer 0
        self.ef_build = ef_construction # how wide the search is while inserting
        # mL controls how quickly the number of nodes per layer shrinks.
        # 1/ln(M) is the standard choice from the original HNSW paper.
        self.mL = 1.0 / math.log(m)

        self.graph: Dict[int, _HNSWNode] = {}
        self.top_layer = -1
        self.entry_point: Optional[int] = None
        self._rng = random.Random(seed)  # seeded so runs are reproducible

    # ------------------------------------------------------------------
    # Random level assignment
    # ------------------------------------------------------------------

    def _random_level(self) -> int:
        """
        Exponential decay: level 0 is most common, level 1 is rarer,
        level 2 rarer still, etc. This is what makes the upper layers
        naturally sparse without any bookkeeping.
        """
        u = max(self._rng.random(), 1e-12)  # avoid log(0)
        return int(math.floor(-math.log(u) * self.mL))

    # ------------------------------------------------------------------
    # Core primitive: best-first search within a single layer
    # ------------------------------------------------------------------

    def _search_layer(
        self, q: np.ndarray, entry_id: int, ef: int, layer: int, dist_fn: DistFn
    ) -> List[Tuple[float, int]]:
        """
        Beam search within one layer of the graph, starting from
        entry_id. Explores outward via graph edges (never touches
        nodes that aren't neighbours-of-neighbours-of... the entry
        point) and keeps the `ef` best candidates found so far.

        `candidates` is a min-heap (nodes still to explore, closest
        first). `found` is a bounded max-heap of the best results seen
        (worst on top, size capped at ef) — same negation trick as
        KDTree's heap.
        """
        visited = {entry_id}
        d0 = dist_fn(q, self.graph[entry_id].item.emb)
        candidates: List[Tuple[float, int]] = [(d0, entry_id)]
        found: List[Tuple[float, int]] = [(-d0, entry_id)]

        while candidates:
            cand_dist, cand_id = heapq.heappop(candidates)
            worst_found = -found[0][0]
            if len(found) >= ef and cand_dist > worst_found:
                break  # nothing left in the queue can beat our worst kept result

            node = self.graph.get(cand_id)
            if node is None or layer >= len(node.neighbors):
                continue

            for nbr_id in node.neighbors[layer]:
                if nbr_id in visited or nbr_id not in self.graph:
                    continue
                visited.add(nbr_id)
                nbr_dist = dist_fn(q, self.graph[nbr_id].item.emb)
                worst_found = -found[0][0] if found else float("inf")
                if len(found) < ef or nbr_dist < worst_found:
                    heapq.heappush(candidates, (nbr_dist, nbr_id))
                    heapq.heappush(found, (-nbr_dist, nbr_id))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = [(-neg_d, item_id) for neg_d, item_id in found]
        result.sort(key=lambda pair: pair[0])
        return result

    @staticmethod
    def _select_neighbors(candidates: List[Tuple[float, int]], max_m: int) -> List[int]:
        """Simple heuristic: just keep the max_m closest. (The original
        HNSW paper also offers a fancier diversity-aware heuristic;
        this project uses the simple version, same as the C++ code.)"""
        return [item_id for _, item_id in candidates[:max_m]]

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, item: VectorItem, dist_fn: DistFn) -> None:
        item_id = item.id
        level = self._random_level()
        self.graph[item_id] = _HNSWNode(item, level)

        if self.entry_point is None:
            # First node ever inserted becomes the entry point.
            self.entry_point = item_id
            self.top_layer = level
            return

        # Phase 1: greedily descend from the top layer down to
        # (level + 1), always moving to whichever neighbour is closest,
        # just to find a good starting point ("entry point") for the
        # layers we'll actually attach to.
        ep = self.entry_point
        for layer in range(self.top_layer, level, -1):
            if layer < len(self.graph[ep].neighbors):
                w = self._search_layer(item.emb, ep, ef=1, layer=layer, dist_fn=dist_fn)
                if w:
                    ep = w[0][1]

        # Phase 2: from min(top_layer, level) down to 0, actually
        # search this layer properly (ef_build wide) and wire up
        # bidirectional edges to the best candidates found.
        for layer in range(min(self.top_layer, level), -1, -1):
            w = self._search_layer(item.emb, ep, ef=self.ef_build, layer=layer, dist_fn=dist_fn)
            max_m = self.M0 if layer == 0 else self.M
            selected = self._select_neighbors(w, max_m)
            self.graph[item_id].neighbors[layer] = selected

            # Wire the edge back from each selected neighbour to us too
            # (edges are bidirectional), then trim that neighbour's
            # edge list back down to max_m if we just overcrowded it —
            # keeping only its truly closest connections.
            for nbr_id in selected:
                neighbor = self.graph.get(nbr_id)
                if neighbor is None:
                    continue
                if len(neighbor.neighbors) <= layer:
                    neighbor.neighbors.extend([] for _ in range(layer + 1 - len(neighbor.neighbors)))
                conn = neighbor.neighbors[layer]
                conn.append(item_id)
                if len(conn) > max_m:
                    scored = sorted(
                        (
                            (dist_fn(neighbor.item.emb, self.graph[c].item.emb), c)
                            for c in conn
                            if c in self.graph
                        ),
                        key=lambda pair: pair[0],
                    )
                    neighbor.neighbors[layer] = [c for _, c in scored[:max_m]]

            if w:
                ep = w[0][1]

        if level > self.top_layer:
            self.top_layer = level
            self.entry_point = item_id

    # ------------------------------------------------------------------
    # k-Nearest-Neighbour search
    # ------------------------------------------------------------------

    def knn(self, q: np.ndarray, k: int, ef: int, dist_fn: DistFn) -> List[Tuple[float, int]]:
        if self.entry_point is None:
            return []

        # Same greedy descent as insert's Phase 1: zoom in via the
        # sparse upper layers first.
        ep = self.entry_point
        for layer in range(self.top_layer, 0, -1):
            if layer < len(self.graph[ep].neighbors):
                w = self._search_layer(q, ep, ef=1, layer=layer, dist_fn=dist_fn)
                if w:
                    ep = w[0][1]

        # Final, wider search at layer 0 (the layer that has every node).
        w = self._search_layer(q, ep, ef=max(ef, k), layer=0, dist_fn=dist_fn)
        return w[:k]

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------

    def remove(self, item_id: int) -> None:
        if item_id not in self.graph:
            return
        # Scrub every edge pointing at this node from every other node.
        for node in self.graph.values():
            for layer_edges in node.neighbors:
                if item_id in layer_edges:
                    layer_edges.remove(item_id)
        if self.entry_point == item_id:
            self.entry_point = None
            for other_id in self.graph:
                if other_id != item_id:
                    self.entry_point = other_id
                    break
        del self.graph[item_id]

    # ------------------------------------------------------------------
    # Introspection (powers the /hnsw-info endpoint + graph visualizer)
    # ------------------------------------------------------------------

    def get_info(self) -> dict:
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes = []
        edges = []
        for node_id, node in self.graph.items():
            nodes.append(
                {
                    "id": node_id,
                    "metadata": node.item.metadata,
                    "category": node.item.category,
                    "maxLyr": node.max_layer,
                }
            )
            top = min(node.max_layer, max_l - 1)
            for layer in range(0, top + 1):
                nodes_per_layer[layer] += 1
                if layer < len(node.neighbors):
                    for nbr_id in node.neighbors[layer]:
                        # Count each undirected edge only once, from the
                        # lower id's perspective.
                        if node_id < nbr_id:
                            edges_per_layer[layer] += 1
                            edges.append({"src": node_id, "dst": nbr_id, "lyr": layer})

        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.graph),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges,
        }

    def size(self) -> int:
        return len(self.graph)
