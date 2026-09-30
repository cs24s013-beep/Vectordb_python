# VectorDB (Python) — Build a Vector Database from Scratch

A line-by-line Python port of the original C++ VectorDB project. Same three
search algorithms (**HNSW**, **KD-Tree**, **Brute Force**), same three
distance metrics, same RAG pipeline over a local Ollama model, same REST
API, same frontend (`index.html` is copied over completely unchanged — it
just talks to whatever server answers on `localhost:8080`).

This README explains **every concept and every file** in enough detail
that you don't need to already know vector databases, Flask, or these
algorithms to follow along. If you already read the C++ version's README,
skip to [What's Different From the C++ Version](#whats-different-from-the-c-version).

---

## Table of Contents

1. [Setup (Windows / Mac / Linux)](#setup)
2. [Running It](#running-it)
3. [Project Structure](#project-structure)
4. [Concept 1 — Vectors & Embeddings](#concept-1--vectors--embeddings)
5. [Concept 2 — Distance Metrics](#concept-2--distance-metrics)
6. [Concept 3 — Brute Force Search](#concept-3--brute-force-search)
7. [Concept 4 — KD-Tree](#concept-4--kd-tree)
8. [Concept 5 — HNSW](#concept-5--hnsw-hierarchical-navigable-small-world)
9. [Concept 6 — Flask & REST APIs](#concept-6--flask--rest-apis)
10. [Concept 7 — Threading & Locks](#concept-7--threading--locks)
11. [Concept 8 — RAG & Ollama](#concept-8--rag--ollama)
12. [File-by-File Walkthrough](#file-by-file-walkthrough)
13. [REST API Reference](#rest-api-reference)
14. [Using the Application](#using-the-application)
15. [Testing Without Installing Ollama](#testing-without-installing-ollama)
16. [Performance Notes (Measured, Not Guessed)](#performance-notes-measured-not-guessed)
17. [What's Different From the C++ Version](#whats-different-from-the-c-version)
18. [Common Issues](#common-issues)
19. [Ideas for Extending This Project](#ideas-for-extending-this-project)

---

## Setup

You need **Python 3.9+** and, if you want the real RAG tab working,
**Ollama**. That's it — no compiler, no PATH surgery.

### Step 1 — Install Python

Most Mac/Linux systems already have it. On Windows, install from
[python.org](https://www.python.org/downloads/) and make sure you tick
**"Add python.exe to PATH"** during install. Verify in a terminal:

```
python --version
```

(On some systems the command is `python3` instead of `python` — use
whichever one prints a 3.x version.)

### Step 2 — Create a virtual environment

A virtual environment is just an isolated folder of installed packages,
so this project's dependencies don't clash with anything else on your
machine. This replaces the "install MSYS2, edit your PATH" step from the
C++ version entirely.

```bash
cd vectordb-python
python -m venv venv

# Activate it:
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows (PowerShell or cmd)
```

You'll know it worked because your terminal prompt gets a `(venv)`
prefix. You need to re-run the activate command every time you open a
new terminal for this project (until you deactivate or close the
terminal).

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs three packages:

| Package | What it's for |
|---|---|
| `flask` | The web server / REST API framework (replaces `httplib.h`) |
| `numpy` | Fast vector math (replaces the hand-written C++ loops) |
| `requests` | HTTP client for talking to Ollama (replaces the C++ `OllamaClient`'s raw sockets) |

### Step 4 — Install Ollama (optional, only needed for Tabs 2 & 3)

1. Go to **[ollama.com](https://ollama.com)** and download it for your OS.
2. Run the installer.
3. Pull the two models this project uses:

```bash
ollama pull nomic-embed-text   # ~274 MB — turns text into vectors
ollama pull llama3.2           # ~2 GB   — generates RAG answers
```

4. Verify:

```bash
ollama list
```

Tab 1 (the demo vector search) works with **no Ollama at all** — it's
fully self-contained with the 20 pre-loaded 16D vectors.

---

## Running It

```bash
python server.py
```

You should see:

```
=== VectorDB Engine (Python) ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

Then open **http://localhost:8080** in your browser. That's the whole
setup — no separate frontend server, no build step. Flask serves
`index.html` directly (see the `/` route in `server.py`).

---

## Project Structure

```
vectordb-python/
├── server.py              ← Flask app: every REST endpoint lives here
├── index.html             ← Frontend, copied unchanged from the C++ project
├── mock_ollama.py         ← Fake Ollama for testing without the real thing
├── requirements.txt       ← pip dependencies
├── README.md              ← This file
└── vectordb/              ← The actual "database" — a Python package
    ├── __init__.py
    ├── models.py           - VectorItem: the one struct every index shares
    ├── distances.py        - cosine / euclidean / manhattan
    ├── brute_force.py      - BruteForce class
    ├── kdtree.py           - KDTree class
    ├── hnsw.py             - HNSW class (the interesting one)
    ├── vector_db.py        - VectorDB: wraps all 3 algorithms for the 16D demo
    ├── document_db.py      - DocumentDB: HNSW-only index for 768D Ollama embeddings
    ├── ollama_client.py    - OllamaClient: talks to Ollama's REST API
    ├── chunker.py          - splits long documents into overlapping chunks
    └── demo_data.py        - the 20 pre-loaded demo vectors
```

Splitting the C++ project's single 1,090-line `main.cpp` into one file
per concept is a very normal Python convention (and honestly would have
been good C++ style too) — it makes each idea easy to find and read in
isolation, which matters a lot for a learning project.

---

## Concept 1 — Vectors & Embeddings

An **embedding** is just a list of numbers (a vector) that represents
the *meaning* of something — usually text, but could be an image, a
song, anything. The key property that makes embeddings useful: things
that mean similar things end up as vectors that are close together in
space.

- In this project's **demo tab**, the 16-dimensional vectors are
  hand-crafted (see `demo_data.py`) so that CS-related items have high
  values in dimensions 0-3, Math-related items have high values in
  dimensions 4-7, and so on. It's a simplified, fake stand-in for a real
  embedding model, built specifically so the four categories cluster
  visibly on the 2D scatter plot.
- In the **Documents tab**, the vectors are *real* — Ollama's
  `nomic-embed-text` model reads your pasted text and outputs a genuine
  768-dimensional semantic embedding. Two chunks of text that mean
  similar things (even if they don't share any of the same words) will
  land close together in that 768-dimensional space.

"Searching" a vector database always means the same thing: given a
query vector, find the k stored vectors that are closest to it. The
three algorithms in this project (BruteForce, KDTree, HNSW) are three
different strategies for doing that search — same question, different
speed/accuracy tradeoffs.

---

## Concept 2 — Distance Metrics

"Closest" needs a definition — that's what a distance metric is. This
project supports three (see `vectordb/distances.py`):

**Euclidean** — straight-line distance, like measuring with a ruler:
```
sqrt( (a1-b1)^2 + (a2-b2)^2 + ... )
```

**Manhattan** — sum of absolute differences per dimension, like a taxi
that can only drive along a city grid, not diagonally:
```
|a1-b1| + |a2-b2| + ...
```

**Cosine** — measures the *angle* between two vectors, ignoring their
length entirely:
```
1 - ( dot(a,b) / (|a| * |b|) )
```
This is the standard choice for text embeddings, because what makes two
sentences "similar" is that they point in a similar direction in
embedding space — not that their raw vectors happen to have the same
magnitude. That's why `document_db.py` hard-codes cosine for the RAG
pipeline, while the demo tab lets you pick any of the three so you can
see how the choice of metric changes which results come back.

---

## Concept 3 — Brute Force Search

The simplest possible answer to "find the k nearest vectors": compute
the distance from the query to **every single stored vector**, sort by
distance, keep the smallest k.

```python
scored = [(dist_fn(q, item.emb), item.id) for item in self.items]
scored.sort(key=lambda pair: pair[0])
return scored[:k]
```

This is always exactly correct (100% recall), because it never skips
anything. Its cost is `O(N × d)` — for N stored vectors of dimension d,
you do N distance calculations, each one costing d operations. Every
other algorithm in this project exists purely to avoid doing this full
scan while still getting (approximately) the same answer. It's also the
baseline the `/benchmark` endpoint measures everything else against.

---

## Concept 4 — KD-Tree

A KD-Tree ("K-Dimensional Tree") is a binary tree that recursively
splits vector space in half, one dimension at a time, cycling through
dimensions as you go deeper (dimension 0 at depth 0, dimension 1 at
depth 1, ... wrapping back around).

**Building it** is like inserting into a normal binary search tree, but
which dimension you compare on changes at every level:

```python
axis = depth % dims
if item.emb[axis] < node.item.emb[axis]:
    go left
else:
    go right
```

**Searching it** has two phases:
1. Walk down to the leaf the query point *would* live at, keeping track
   of the best k candidates seen so far in a small heap.
2. Walk back **up**, and at each node ask: *"could the other half of
   this split possibly contain something closer than my current worst
   candidate?"* If the query's distance to the splitting line is
   already farther than the worst candidate you're holding, that whole
   subtree is skipped — pruned — because nothing in it can beat what
   you already have.

This pruning is what makes KD-Tree fast in **low** dimensions: each
level of the tree throws away roughly half the remaining candidates,
similar to binary search. It breaks down in **high** dimensions
(~50+) because in high-dimensional space, almost every point ends up
close to *some* splitting boundary — the "could the other side be
closer?" check keeps saying yes, nothing gets pruned, and you're back
to a full scan with extra tree-traversal overhead on top. That's the
textbook reason this project reaches for HNSW, not KD-Tree, for the
768-dimensional real document embeddings (see the honest measured
numbers in [Performance Notes](#performance-notes-measured-not-guessed)
though — the theory and small-scale reality don't always match).

Deletion is the one place KD-Tree is awkward: removing a node from the
middle of the tree can break its splitting invariants, so this project
does what the original C++ does too — throw the whole tree away and
rebuild it from whatever's left (`KDTree.rebuild`). Fine for a
demo-sized dataset; a production KD-Tree would use lazy/soft deletion.

---

## Concept 5 — HNSW (Hierarchical Navigable Small World)

This is the algorithm real production vector databases (Pinecone,
Weaviate, Chroma, Milvus, Qdrant) actually use. It's *approximate*
nearest-neighbour search — it can occasionally miss the true best
match — in exchange for search speed that barely slows down as a
dataset grows into the millions.

### The subway-map mental model

Every stored vector is a node in a graph, connected to its approximate
nearest neighbours. But instead of one flat graph, HNSW builds several
stacked **layers**:

```
Layer 2 (very few nodes)     o---------------o        <- express lines,
Layer 1 (some nodes)         o---o-------o----o           long jumps
Layer 0 (every node)     o-o-o-o-o-o-o-o-o-o-o-o-o     <- local stops,
                                                            every node
```

Each node is randomly assigned a "max layer" when it's inserted, with
higher layers being exponentially rarer (`HNSW._random_level` — most
nodes only ever exist at layer 0). It's exactly like a real subway
map: express trains (upper layers) skip most stops so you cover a lot
of ground fast, then you switch to the local line (layer 0) to reach
your exact destination.

### Searching

1. Start at the single entry point on the **topmost** layer.
2. Greedily walk toward the query — at each step, look at the current
   node's neighbours *on this layer* and move to whichever one is
   closest to the query. Keep doing this until you can't improve.
3. Drop down one layer, using where you ended up as the new starting
   point. Repeat.
4. Once you reach layer 0 (every node lives here), do a slightly wider
   search — instead of tracking just the single best node, track the
   best `ef` candidates — and return the top k of those.

### Inserting

Same greedy descent as searching, used to find where a new node should
attach, then: at every layer from the new node's assigned max layer
down to 0, connect it to its `M` nearest neighbours (bidirectionally).
If adding this new edge makes some existing neighbour too crowded
(more than `M` connections), trim that neighbour's connection list back
down to its truly-closest `M` neighbours.

```python
# vectordb/hnsw.py, simplified
for layer in range(min(self.top_layer, level), -1, -1):
    candidates = self._search_layer(item.emb, ep, ef=self.ef_build, layer=layer, dist_fn=dist_fn)
    selected = self._select_neighbors(candidates, max_m)
    self.graph[item_id].neighbors[layer] = selected
    # ...then wire the reverse edges, trimming overcrowded neighbours
```

### Why it's fast

Most of the "distance" between the entry point and your query — measured
in graph hops, not vector-space distance — gets covered by a handful of
long-range jumps on the sparse upper layers. You never end up looking
at more than a small, roughly constant number of nodes, no matter how
many millions are stored. Compare that to Brute Force, which must look
at literally everything, every single time.

### The min-heap / max-heap trick

Python's built-in `heapq` module only implements a **min-heap** (the
smallest item is always on top). But both KD-Tree and HNSW need a
**bounded max-heap** — one that keeps the k *smallest* distances seen so
far, with the *worst* of those k sitting on top so it's the first thing
evicted when something better shows up. The trick used throughout
`kdtree.py` and `hnsw.py`:

```python
heapq.heappush(heap, (-distance, item_id))   # negate on the way in
...
worst_kept_distance = -heap[0][0]            # negate again on the way out
```

Negating a number flips a min-heap into behaving like a max-heap. This
directly replaces C++'s `std::priority_queue`, which is a max-heap by
default.

---

## Concept 6 — Flask & REST APIs

**Flask** is a Python web framework: you write a normal Python function,
decorate it with the URL it should handle, and Flask takes care of
listening on a socket, parsing HTTP requests, and calling your function.

```python
@app.route("/search", methods=["GET"])
def search():
    q = parse_vec(request.args.get("v", ""))   # ?v=0.1,0.2,...
    ...
    return jsonify({"results": [...]})          # auto-converts dict -> JSON HTTP response
```

Compare that to the C++ version's equivalent, which needed the entire
20,000-line `httplib.h` just to get an HTTP server, plus hand-rolled
functions (`jS`, `jVec`, `extractStr`, `parseBody`) to build and parse
JSON one character at a time because `httplib` has no JSON support.
Flask plus Python's built-in `json` module (which `request.get_json()`
and `jsonify()` use internally) collapses all of that into a few lines
per endpoint.

A few Flask idioms used throughout `server.py`:

| Code | Meaning |
|---|---|
| `@app.route("/delete/<int:item_id>", methods=["DELETE"])` | `item_id` is pulled straight out of the URL and converted to `int` automatically |
| `request.args.get("k", 5)` | Read query-string parameter `?k=...`, defaulting to `5` if absent |
| `request.get_json(silent=True)` | Parse a JSON request body into a dict; `silent=True` returns `None` instead of raising if the body isn't valid JSON |
| `jsonify({...})` | Convert a Python dict into a proper JSON HTTP response with the right `Content-Type` header |
| `@app.after_request` | A function that runs on **every** response, used here to attach CORS headers |

### CORS, briefly

Browsers block JavaScript from calling a server on a different
origin (scheme + host + port) unless that server explicitly says it's
okay, via `Access-Control-Allow-Origin` and friends. The
`add_cors_headers` function in `server.py` attaches those headers to
every response Flask sends — including Flask's automatic replies to the
`OPTIONS` "preflight" requests browsers send before certain
cross-origin calls — which is all the CORS handling this project needs.

---

## Concept 7 — Threading & Locks

Flask's dev server, run with `threaded=True`, can handle multiple
requests **at the same time**, on different threads. That's necessary
here because `/doc/ask` can take many seconds waiting on the LLM — you
don't want that one slow request to freeze every other request in the
meantime.

But it introduces a real danger: if two requests both try to insert
into the same Python dictionary or graph structure at the exact same
instant, you can corrupt it (e.g. one thread's write gets partially
overwritten by another's). `VectorDB` and `DocumentDB` both guard every
method with a `threading.Lock`:

```python
def insert(self, ...):
    with self._lock:
        # only one thread at a time can be inside this block
        ...
```

This is the direct Python equivalent of `std::mutex` /
`std::lock_guard` in the original C++ — same problem, same solution,
different syntax.

---

## Concept 8 — RAG & Ollama

**RAG** stands for Retrieval-Augmented Generation: instead of asking an
LLM a question and hoping it happens to know the answer from its
training data, you first *retrieve* relevant passages from your own
documents, then hand those passages to the LLM as context alongside the
question. This is how you get an LLM to answer questions about content
it was never trained on (your notes, your company's docs, whatever you
pasted into Tab 2).

The full pipeline, implemented in the `/doc/ask` endpoint:

```
1. question  ──[Ollama embeds it]──▶  768D query vector
2. query vector ──[HNSW search]──▶  top-k most similar document chunks
3. chunks + question ──[assembled into a prompt]──▶  sent to llama3.2
4. llama3.2 ──[reads the prompt]──▶  generates an answer grounded in your docs
```

**Ollama** is a program that runs LLMs entirely on your own machine —
nothing is sent to the cloud. It exposes a simple local REST API on
port 11434. This project's `OllamaClient` (in `vectordb/ollama_client.py`)
calls two of its endpoints:

- `POST /api/embeddings` — turn text into a vector (used for both
  indexing documents and embedding questions)
- `POST /api/generate` — turn a prompt into a generated answer

Before embedding, long documents get split into overlapping ~250-word
chunks (`vectordb/chunker.py`). Two reasons: embedding models work
better on focused passages than on entire documents at once, and
overlap prevents a sentence that happens to land right on a chunk
boundary from being cut in half and losing its meaning in both
resulting pieces.

---

## File-by-File Walkthrough

**`vectordb/models.py`** — `VectorItem`, the one shared struct
(id, metadata, category, embedding) every index stores and returns. In
its own file to avoid a circular import between `vector_db.py` and
`document_db.py`.

**`vectordb/distances.py`** — the three distance functions, plus
`get_dist_fn(name)` to look one up by string name (used because the
frontend sends the metric as a query-string value like `?metric=cosine`).

**`vectordb/brute_force.py`** — `BruteForce`: a plain list of items,
`knn()` scores every one and returns the smallest k.

**`vectordb/kdtree.py`** — `KDTree`: recursive `_insert`/`_knn` with the
axis-cycling split and the "prune the far side if it can't possibly
win" check described in [Concept 4](#concept-4--kd-tree).

**`vectordb/hnsw.py`** — `HNSW`: the multilayer graph described in
[Concept 5](#concept-5--hnsw-hierarchical-navigable-small-world).
`_search_layer` is the core primitive (a bounded best-first search
within one layer); `insert` and `knn` both call it repeatedly while
walking down through the layers.

**`vectordb/vector_db.py`** — `VectorDB`: the "front desk" for the 16D
demo dataset. Owns one of each index (BruteForce, KDTree, HNSW), keeps
all three in sync on every insert/delete, and lets the caller pick
which one answers a given `search()` call — that's how the UI's
"compare all algorithms" button works: same data, three engines, one
call each. Guards everything with a `threading.Lock`.

**`vectordb/document_db.py`** — `DocumentDB`: same front-desk idea, but
for real 768D Ollama embeddings, and HNSW-only (no KD-Tree — see
Concept 4 for why) plus a BruteForce fallback for very small document
sets. Distance metric is hard-coded to cosine.

**`vectordb/ollama_client.py`** — `OllamaClient`: wraps Ollama's two
REST endpoints (`/api/embeddings`, `/api/generate`) plus an
`is_available()` health check.

**`vectordb/chunker.py`** — `chunk_text()`: splits long text into
overlapping ~250-word windows before embedding.

**`vectordb/demo_data.py`** — the 20 hand-crafted 16D vectors (5 each
across CS / Math / Food / Sports) that get loaded at startup, plus
`load_demo(db)` which inserts them all.

**`server.py`** — the Flask app. Creates one `VectorDB`, one
`DocumentDB`, one `OllamaClient`; defines every REST endpoint as a
`@app.route`-decorated function; serves `index.html` at `/`; starts the
dev server on port 8080 when run directly.

**`mock_ollama.py`** — a fake Ollama server for testing the RAG
pipeline without installing the real (multi-gigabyte) models. See
[Testing Without Installing Ollama](#testing-without-installing-ollama).

---

## REST API Reference

Identical to the C++ version — this is exactly why `index.html` didn't
need to change at all.

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure and layer stats |
| `GET` | `/stats` | Database statistics |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store document |
| `GET` | `/doc/list` | — | List all stored documents |
| `DELETE` | `/doc/delete/:id` | — | Delete document chunk |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| `GET` | `/status` | — | Ollama status and model info |

### Example: search via curl

```bash
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

### Example: ask a question via curl

```bash
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'
```

---

## Using the Application

### Tab 1: Search (Demo Vectors)

Type a concept (`binary tree`, `sushi`, `basketball`, `calculus`), pick
an algorithm and a distance metric, hit **⚡ SEARCH**. The scatter plot
projects all 20 vectors to 2D via PCA (computed entirely client-side in
`index.html` — the Python server never sees this step) — watch how the
four semantic categories form visually distinct clusters. **▶ COMPARE
ALL ALGOS** runs all three algorithms on the same query and shows their
relative speed.

### Tab 2: Documents (Real Embeddings)

Paste any text (lecture notes, articles, whatever). It gets chunked,
each chunk embedded by Ollama into a real 768D vector, and stored in
the document HNSW index.

### Tab 3: Ask AI (RAG Pipeline)

Insert some documents first, then ask a question about them. Behind the
scenes: your question is embedded, HNSW retrieves the 3 most similar
chunks, those chunks are stuffed into a prompt as context, and
`llama3.2` generates an answer grounded in that context. Click the
context chips to see exactly which chunks the model was given.

---

## Testing Without Installing Ollama

Real Ollama needs ~2.3 GB of model downloads before Tabs 2 & 3 do
anything. If you just want to confirm the whole pipeline (server,
chunking, HNSW indexing, RAG prompt assembly) actually works before
committing to that download, use the included fake:

```bash
# Terminal 1
python mock_ollama.py

# Terminal 2
python server.py

# Terminal 3 — or just use the browser at localhost:8080
curl -X POST http://localhost:8080/doc/insert -H "Content-Type: application/json" \
  -d '{"title":"Test","text":"Some text to embed and store."}'
```

`mock_ollama.py` fakes Ollama's three endpoints well enough for
`server.py` to treat it as the real thing — it just returns a
deterministic fake embedding (a hash of the input text) instead of a
real semantic one, and a canned string instead of a real LLM answer.
Good for proving the plumbing works; not useful for judging whether RAG
answers are actually *good*, since the fake embeddings carry no real
meaning.

---

## Performance Notes (Measured, Not Guessed)

It would be easy to just repeat "HNSW is O(log N), so it's always
fastest" — but that's a claim about *asymptotic* behaviour as N gets
very large, not a guarantee about wall-clock time on any dataset size
you happen to test. Here's what actually happened when this Python port
was benchmarked (your numbers will vary by machine, but the *pattern*
is worth understanding):

**3,000 vectors, 32 dimensions, cosine distance, 20 random queries:**

| Algorithm | Avg latency |
|---|---|
| Brute Force | ~10,700 µs |
| KD-Tree | ~410 µs |
| HNSW | ~3,470 µs |

**800 vectors, 768 dimensions (Ollama's real embedding size), same test:**

| Algorithm | Avg latency |
|---|---|
| Brute Force | ~2,820 µs |
| KD-Tree | ~430 µs |
| HNSW | ~2,280 µs |

Notice KD-Tree *wins* both times, and HNSW is barely faster than brute
force at these sizes (HNSW's recall — how often it agreed with brute
force's exact answer — was 97-99%, so it's doing real work, just not
winning the speed race yet). Two honest reasons why, worth understanding
rather than glossing over:

1. **Dataset size matters more than people expect.** HNSW's real
   advantage only shows up once brute force would have to scan
   *tens of thousands to millions* of vectors. At a few hundred or
   thousand, the constant overhead of maintaining HNSW's graph
   structure (heap operations, dictionary lookups, bookkeeping) can
   outweigh the benefit of skipping most of the dataset — there just
   isn't much dataset to skip yet.
2. **Pure-Python function-call overhead dominates at small scale.**
   Every distance calculation here goes through a Python function call
   plus a numpy call, and numpy's own overhead per call is significant
   relative to the tiny amount of actual math being done on a 16-768
   element vector. Compiled C++ doesn't pay this tax, so the original
   project's relative speed differences between algorithms look
   different from this Python port's, especially at small N.

The algorithmic *shape* of the tradeoffs (why HNSW scales better
asymptotically, why KD-Tree degrades in high dimensions as N grows
large) is still completely real and worth learning — it's just not
something you can always see in a quick local benchmark on a few
thousand items. If you want to see HNSW pull ahead for real, try
inserting 50,000+ vectors and compare again; you should see the gap
start to open up. That experiment — and reasoning about *why* the
numbers look the way they do — is arguably a better way to internalize
Big-O complexity than reading the definition again.

---

## What's Different From the C++ Version

| | C++ | Python |
|---|---|---|
| HTTP server | `httplib.h` (~20,000 lines, single header) | Flask (~10 lines to start a server) |
| JSON | Hand-rolled string scanning (`extractStr`, `parseBody`, `jS`, `jVec`) | Built-in `json` module via `request.get_json()` / `jsonify()` |
| Vector math | Manual `for` loops | `numpy` vectorized operations |
| Ollama HTTP calls | Raw `httplib::Client`, manual JSON building | `requests.post(..., json={...})` |
| Concurrency | `std::mutex` / `std::lock_guard` | `threading.Lock` |
| Priority queues | `std::priority_queue` (max-heap by default) | `heapq` (min-heap; negate values to flip it) |
| Compilation | `g++ -std=c++17 -O2 main.cpp -o db -lws2_32` | None — just run the script |
| Setup | MSYS2 + PATH editing (Windows-specific) | `pip install -r requirements.txt` (same everywhere) |
| Frontend | `index.html`, served from a hand-written route | `index.html`, unchanged, served by Flask's `send_from_directory` |

The algorithms themselves (BruteForce, KDTree, HNSW) are functionally
identical — same complexity, same behavior, same REST contract — so
this port is a genuinely fair way to compare "the same project in two
languages" rather than a different project that happens to do similar
things.

---

## Common Issues

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'flask'` | You forgot to activate the virtual environment, or forgot `pip install -r requirements.txt` |
| `Address already in use` on port 8080 | Something else is already listening there. Mac/Linux: `lsof -i :8080` then `kill <pid>`. Windows: `netstat -ano \| findstr 8080` then `taskkill /PID <pid> /F` |
| `Ollama: OFFLINE` in the startup banner | Run `ollama serve` in another terminal, or use `mock_ollama.py` to test without it |
| Embedding takes forever the first time | Ollama is downloading the model on first use — wait a couple of minutes |
| RAG answers are slow | Normal — `llama3.2` on a laptop CPU takes 10-30s. Try `ollama pull llama3.2:1b` and change `self.gen_model` in `vectordb/ollama_client.py` for a faster, smaller model |
| `/doc/ask` always says "Ollama unavailable" | Check `curl http://localhost:11434/api/tags` works at all — if not, Ollama itself isn't running |

---

## Ideas for Extending This Project

If you want to keep learning by modifying this codebase rather than
just reading it:

- **Persistence.** Right now everything lives in memory and disappears
  when you stop the server. Try serializing `VectorDB`/`DocumentDB` to
  a JSON or pickle file on shutdown and reloading on startup.
- **A real recall benchmark endpoint.** Add a `/recall-test` route that
  runs N random queries through both HNSW and BruteForce and reports
  what percentage of results matched — turn the manual experiment from
  [Performance Notes](#performance-notes-measured-not-guessed) into a
  button in the UI.
- **Swap in `numpy`'s batch operations for BruteForce.** Instead of a
  Python loop calling `dist_fn` once per item, compute all distances in
  one vectorized numpy call (`np.linalg.norm(all_embeddings - q, axis=1)`
  for euclidean). This alone should make BruteForce dramatically faster
  and is a great concrete lesson in why "vectorization" is such a
  common piece of performance advice in Python.
- **Streaming RAG answers.** Ollama's `/api/generate` supports
  `"stream": true` — try changing `OllamaClient.generate` to stream
  tokens back to the browser via
  [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
  for a real (not faked) typewriter effect.
- **A fourth algorithm.** Try implementing IVF (Inverted File Index,
  the other major approach real vector databases use alongside HNSW)
  and adding it as a fourth option next to bruteforce/kdtree/hnsw.

---

## License

MIT — use this however you want.
