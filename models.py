"""
Shared data model.

VectorItem is the one struct every index (BruteForce, KDTree, HNSW)
stores and returns. Keeping it in its own file avoids a circular import
between vector_db.py and document_db.py, which both need it.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: np.ndarray
