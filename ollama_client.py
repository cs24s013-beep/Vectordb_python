"""
OllamaClient
=============
A thin wrapper around Ollama's local REST API (Ollama runs on your own
machine at http://127.0.0.1:11434 — nothing leaves your computer).

This replaces the original C++ OllamaClient class, which had to
hand-build raw JSON strings and parse raw JSON responses with manual
string scanning because httplib.h has no JSON support at all. Python's
`requests` + built-in `json` module (which `requests` uses internally
for both directions) reduces all of that to two straightforward calls.

Two Ollama endpoints are used:
  POST /api/embeddings  -> turn text into a vector (for search/RAG)
  POST /api/generate    -> turn a prompt into a text answer (for RAG)
"""

from typing import List
import requests


class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.base_url = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def is_available(self) -> bool:
        """Quick health check used at startup and by /status."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def embed(self, text: str) -> List[float]:
        """
        Returns the embedding vector for `text`, or [] if Ollama isn't
        running / the model isn't pulled. Callers treat an empty list
        as "Ollama unavailable" (same contract as the C++ version,
        which returned an empty std::vector<float> for the same case).
        """
        try:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=30,
            )
            if resp.status_code != 200:
                return []
            return resp.json().get("embedding", [])
        except requests.RequestException:
            return []

    def generate(self, prompt: str) -> str:
        """
        Returns the LLM's full response text (stream=False, so we wait
        for the whole answer rather than dealing with a token stream —
        simpler for a learning project, at the cost of no "typing"
        effect from the server itself; the frontend fakes that
        typewriter effect client-side once the full answer arrives).
        """
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=180,  # local LLMs on a laptop CPU can be slow
            )
            if resp.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return resp.json().get("response", "")
        except requests.RequestException:
            return "ERROR: Ollama unavailable. Run: ollama serve"
