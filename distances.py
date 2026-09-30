"""
Distance metrics
=================
Three ways to measure "how far apart" two vectors are. Every search
algorithm in this project (BruteForce, KDTree, HNSW) is generic over
whichever distance function you hand it — that's what DistFn meant in
the original C++ (a std::function). Here it's just "any Python callable
that takes two numpy arrays and returns a float".

All three take numpy arrays so the actual math is vectorized (executed
in C under the hood via numpy) instead of using a Python for-loop like
the original C++ does with a manual loop over indices.
"""

import numpy as np


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """
    Straight-line ("as the crow flies") distance between two points.
    sqrt(sum((a_i - b_i)^2))

    Smaller = more similar. Range: [0, infinity).
    Sensitive to vector magnitude — a vector scaled by 10x looks very
    different even if it points in the same direction.
    """
    return float(np.linalg.norm(a - b))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Measures the ANGLE between two vectors, ignoring their length.
    cosine_similarity = dot(a, b) / (|a| * |b|)   -> ranges [-1, 1]

    We return a *distance*, not a similarity, so smaller must still mean
    "more similar" (to match euclidean/manhattan semantics and let every
    algorithm use "pick the smallest" as its rule). That's why we return
    1 - similarity:
        identical direction      -> similarity  1.0 -> distance 0.0
        orthogonal (unrelated)   -> similarity  0.0 -> distance 1.0
        opposite direction       -> similarity -1.0 -> distance 2.0

    This is the standard metric for text/semantic embeddings (what
    Ollama produces) because two sentences that mean the same thing
    point in a similar direction in embedding space, regardless of
    how "long" the raw embedding vector happens to be.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        # A zero-length vector has no direction, so "angle" is undefined.
        # Treat it as maximally different rather than raising/crashing.
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def manhattan(a: np.ndarray, b: np.ndarray) -> float:
    """
    "Taxicab" distance: sum of absolute differences per dimension.
    sum(|a_i - b_i|)

    Named after the fact that a taxi in Manhattan can't drive diagonally
    through a building — it has to go along the grid, block by block.
    Less sensitive to a single huge outlier dimension than euclidean
    (no squaring), but still magnitude-sensitive like euclidean.
    """
    return float(np.sum(np.abs(a - b)))


def get_dist_fn(name: str):
    """Look up a distance function by name, defaulting to euclidean."""
    if name == "cosine":
        return cosine
    if name == "manhattan":
        return manhattan
    return euclidean
