"""
Text chunking
==============
Embedding models (and LLMs) can only accept so much text at once, and
even when they technically can accept a lot, a shorter, focused chunk
produces a more meaningful embedding than one giant vector trying to
represent an entire document at once. So before embedding, long
documents are split into overlapping windows of ~250 words.

The OVERLAP (30 words by default) matters: without it, a sentence that
happens to fall right at a chunk boundary gets cut in half and neither
resulting chunk fully captures its meaning. Overlapping windows make
sure every sentence appears whole in at least one chunk.
"""

from typing import List


def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step = chunk_words - overlap_words  # how far the window slides each time
    i = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks
