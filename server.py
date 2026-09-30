"""
server.py — the Flask app
===========================
This file replaces main.cpp's httplib server setup. Every endpoint here
is a direct 1:1 port of the corresponding svr.Get/svr.Post/svr.Delete
handler in the original C++ — same URL, same query params / JSON body,
same JSON response shape — so the existing index.html frontend works
completely unchanged against this server.

Flask basics used here, if you're new to the framework:
  @app.route("/path", methods=["GET"])   registers a URL -> function
  request.args.get("k")                  reads a ?k=... query string param
  request.get_json()                     parses a JSON request body into a dict
  jsonify({...})                         turns a dict into a JSON HTTP response
  <int:item_id>                          a URL segment auto-converted to int
                                          (e.g. /delete/5 -> item_id = 5)

Run it with:  python server.py
"""

import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

from vectordb.distances import get_dist_fn
from vectordb.vector_db import VectorDB
from vectordb.document_db import DocumentDB
from vectordb.ollama_client import OllamaClient
from vectordb.chunker import chunk_text
from vectordb.demo_data import load_demo

DIMS = 16  # dimensionality of the demo vectors (independent of Ollama's 768D)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()


# ----------------------------------------------------------------------
# CORS
# ----------------------------------------------------------------------
# The frontend (index.html) is opened as a plain file / served from this
# same Flask app, but in general a browser blocks JavaScript from calling
# a different origin (scheme+host+port) unless that origin explicitly
# allows it via these headers. after_request runs on every single
# response — including Flask's automatic replies to OPTIONS "preflight"
# requests that browsers send before certain cross-origin calls — so this
# one function is all the CORS handling this project needs.
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def parse_vec(raw: str) -> np.ndarray:
    """Parse 'f1,f2,f3,...' from a query string into a numpy array."""
    if not raw:
        return np.array([])
    values = []
    for part in raw.split(","):
        try:
            values.append(float(part))
        except ValueError:
            pass  # silently skip malformed numbers, same as the C++ parseVec
    return np.array(values, dtype=np.float32)


# ========================================================================
#  DEMO VECTOR ENDPOINTS
# ========================================================================

@app.route("/search", methods=["GET"])
def search():
    q = parse_vec(request.args.get("v", ""))
    if q.shape[0] != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"})

    k = int(request.args.get("k", 5) or 5)
    metric = request.args.get("metric") or "cosine"
    algo = request.args.get("algo") or "hnsw"

    out = db.search(q, k, metric, algo)
    results = [
        {
            "id": h["id"],
            "metadata": h["meta"],
            "category": h["cat"],
            "distance": round(float(h["dist"]), 6),
            "embedding": [round(float(x), 4) for x in h["emb"]],
        }
        for h in out["hits"]
    ]
    return jsonify(
        {"results": results, "latencyUs": out["us"], "algo": out["algo"], "metric": out["metric"]}
    )


@app.route("/insert", methods=["POST"])
def insert():
    body = request.get_json(silent=True) or {}
    metadata = body.get("metadata", "")
    category = body.get("category", "")
    embedding = body.get("embedding", [])

    if not metadata or not embedding or len(embedding) != DIMS:
        return jsonify({"error": "invalid body"})

    item_id = db.insert(metadata, category, np.array(embedding, dtype=np.float32), get_dist_fn("cosine"))
    return jsonify({"id": item_id})


@app.route("/delete/<int:item_id>", methods=["DELETE"])
def delete(item_id):
    ok = db.remove(item_id)
    return jsonify({"ok": ok})


@app.route("/items", methods=["GET"])
def items():
    out = [
        {
            "id": v.id,
            "metadata": v.metadata,
            "category": v.category,
            "embedding": [round(float(x), 4) for x in v.emb],
        }
        for v in db.all()
    ]
    return jsonify(out)


@app.route("/benchmark", methods=["GET"])
def benchmark():
    q = parse_vec(request.args.get("v", ""))
    if q.shape[0] != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"})

    k = int(request.args.get("k", 5) or 5)
    metric = request.args.get("metric") or "cosine"
    b = db.benchmark(q, k, metric)
    return jsonify(
        {"bruteforceUs": b["bfUs"], "kdtreeUs": b["kdUs"], "hnswUs": b["hnswUs"], "itemCount": b["n"]}
    )


@app.route("/hnsw-info", methods=["GET"])
def hnsw_info():
    return jsonify(db.hnsw_info())


# ========================================================================
#  DOCUMENT + RAG ENDPOINTS
# ========================================================================

@app.route("/doc/insert", methods=["POST"])
def doc_insert():
    body = request.get_json(silent=True) or {}
    title = body.get("title", "")
    text = body.get("text", "")
    if not title or not text:
        return jsonify({"error": "need title and text"})

    chunks = chunk_text(text, 250, 30)
    ids = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return jsonify(
                {
                    "error": "Ollama unavailable. Install from https://ollama.com then run: "
                    "ollama pull nomic-embed-text && ollama pull llama3.2"
                }
            )
        chunk_title = f"{title} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else title
        ids.append(doc_db.insert(chunk_title, chunk, np.array(emb, dtype=np.float32)))

    return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.dims})


@app.route("/doc/delete/<int:item_id>", methods=["DELETE"])
def doc_delete(item_id):
    ok = doc_db.remove(item_id)
    return jsonify({"ok": ok})


@app.route("/doc/list", methods=["GET"])
def doc_list():
    out = []
    for d in doc_db.all():
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        out.append({"id": d.id, "title": d.title, "preview": preview, "words": d.text.count(" ") + 1})
    return jsonify(out)


@app.route("/doc/search", methods=["POST"])
def doc_search():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    k = int(body.get("k", 3) or 3)
    if not question:
        return jsonify({"error": "need question"})

    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"})

    hits = doc_db.search(np.array(q_emb, dtype=np.float32), k)
    contexts = [{"id": doc.id, "title": doc.title, "distance": round(float(dist), 4)} for dist, doc in hits]
    return jsonify({"contexts": contexts})


@app.route("/doc/ask", methods=["POST"])
def doc_ask():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    k = int(body.get("k", 3) or 3)
    if not question:
        return jsonify({"error": "need question"})

    # Step 1: embed the question
    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"})

    # Step 2: retrieve the top-k most relevant chunks
    hits = doc_db.search(np.array(q_emb, dtype=np.float32), k)

    # Step 3: build the prompt, stuffing retrieved chunks in as context
    ctx = "".join(f"[{i + 1}] {doc.title}:\n{doc.text}\n\n" for i, (_, doc) in enumerate(hits))
    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{ctx}"
        f"Question: {question}\n\n"
        "Answer:"
    )

    # Step 4: generate the answer
    answer = ollama.generate(prompt)

    # Step 5: return everything (answer + the contexts that were used,
    # so the UI can show which chunks the model actually saw)
    contexts = [
        {"id": doc.id, "title": doc.title, "text": doc.text, "distance": round(float(dist), 4)}
        for dist, doc in hits
    ]
    return jsonify({"answer": answer, "model": ollama.gen_model, "contexts": contexts, "docCount": doc_db.size()})


@app.route("/status", methods=["GET"])
def status():
    up = ollama.is_available()
    return jsonify(
        {
            "ollamaAvailable": up,
            "embedModel": ollama.embed_model,
            "genModel": ollama.gen_model,
            "docCount": doc_db.size(),
            "docDims": doc_db.dims,
            "demoDims": DIMS,
            "demoCount": db.size(),
        }
    )


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify(
        {
            "count": db.size(),
            "dims": DIMS,
            "algorithms": ["bruteforce", "kdtree", "hnsw"],
            "metrics": ["euclidean", "cosine", "manhattan"],
        }
    )


# ========================================================================
#  FRONTEND
# ========================================================================

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(BASE_DIR, "index.html")


# ========================================================================
#  ENTRY POINT
# ========================================================================

if __name__ == "__main__":
    load_demo(db)

    ollama_up = ollama.is_available()
    print("=== VectorDB Engine (Python) ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    # threaded=True lets Flask's dev server handle more than one request
    # at a time (important since /doc/ask can take many seconds while
    # waiting on the LLM) — this is why VectorDB/DocumentDB use locks.
    app.run(host="0.0.0.0", port=8080, threaded=True)
