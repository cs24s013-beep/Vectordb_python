"""
mock_ollama.py — a fake Ollama server for testing without the real thing
==========================================================================
Real Ollama requires installing a multi-GB local LLM before you can even
see the RAG pipeline (Tab 3) light up. This script fakes just enough of
Ollama's REST API (/api/tags, /api/embeddings, /api/generate) so you can
verify your whole server + RAG pipeline works BEFORE spending the time
downloading real models.

The "embeddings" it returns are not real semantic embeddings — they're
just a deterministic hash of the input text turned into 32 numbers, so
identical text always gets the identical fake vector. That's enough to
prove insert -> store -> retrieve -> generate all wire together
correctly; it is NOT enough to prove your RAG answers are actually
relevant to the question (only a real embedding model can do that).

Usage:
    Terminal 1:  python mock_ollama.py       (fakes Ollama on port 11434)
    Terminal 2:  python server.py            (your VectorDB server)
    Terminal 3:  curl the /doc/* endpoints, or open http://localhost:8080

Once you've confirmed everything works end-to-end, stop this script,
install real Ollama (see README.md), and run the real thing on the same
port instead — server.py doesn't need to change at all.
"""

import hashlib
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/api/tags", methods=["GET"])
def tags():
    return jsonify({"models": [{"name": "nomic-embed-text"}, {"name": "llama3.2"}]})


@app.route("/api/embeddings", methods=["POST"])
def embeddings():
    text = (request.get_json() or {}).get("prompt", "")
    # Deterministic fake embedding: hash the text into 32 floats in [0, 1).
    digest = hashlib.sha256(text.encode()).digest()
    vector = [b / 255.0 for b in digest]  # 32 dimensions (sha256 = 32 bytes)
    return jsonify({"embedding": vector})


@app.route("/api/generate", methods=["POST"])
def generate():
    prompt = (request.get_json() or {}).get("prompt", "")
    return jsonify({"response": f"[MOCK ANSWER — real Ollama not installed. Prompt was {len(prompt)} chars.]"})


if __name__ == "__main__":
    print("=== Mock Ollama ===")
    print("Faking http://127.0.0.1:11434 (/api/tags, /api/embeddings, /api/generate)")
    print("This is NOT a real LLM — see the docstring at the top of this file.")
    app.run(host="127.0.0.1", port=11434)
