# DevAgent — Work Update

_Last updated: 2026-06-12 (Phase 1 complete; Phase 2 in progress — state, llm, planner done)_

A running log of what's built, what's next, and the phased plan to get from the
current retrieval layer to the full autonomous issue-resolver described in the
README.

---

## TL;DR

DevAgent is being built bottom-up. The **retrieval layer is complete**
(chunker ✅, indexer ✅, retriever ✅). The **agent**, **backend**,
**sandbox**, and **dashboard** layers from the README do not exist yet —
**Phase 2 (agent nodes) is next.**

| Layer | Status |
|---|---|
| Retrieval (chunk → index → retrieve) | ✅ Complete (3 of 3 modules) |
| Agent nodes (LangGraph) | 🟡 In progress (state, llm, planner done) |
| Backend / webhook (FastAPI) | 🔴 Not started |
| GitHub integration | 🔴 Not started |
| Sandbox (Docker exec) | 🔴 Not started |
| Dashboard (Next.js) | 🔴 Not started |
| Evals / benchmark | 🔴 Not started |

**Tests:** 25 passing (`pytest -q`).

---

## Work done so far

### Infrastructure
- **`docker-compose.yml`** — Qdrant (`:6333`) + MLflow (`:5000`) with persistent
  named volumes. Verified Qdrant reachable from Python (`collections=[]`).
- **`requirements.txt`** — retrieval-layer deps pinned (`tree-sitter>=0.24` to
  match the ABI-15 language packs, `sentence-transformers`, `qdrant-client`,
  `python-dotenv`, `pytest`).
- **`.env.example`** — config template (OpenAI/Anthropic keys, GitHub App,
  Qdrant host/port, MLflow URI, `DEVAGENT_INDEX_ROOT`).

### Step 1.1 — `app/retrieval/chunker.py` ✅
- Parses source with **tree-sitter** (Python **and** JavaScript).
- Emits **function / class-level chunks** (not character windows).
- Metadata per chunk: `file_path`, `name`, `node_type`, `start_line`,
  `end_line`, `imports`, and **`calls`** (callee names — the dependency edges
  the retriever will use to expand context).
- Falls back to a single file-level chunk when a file has no top-level
  definitions.

### Step 1.2 — `app/retrieval/indexer.py` ✅
- Clones a repo, walks `.py`/`.js` files (skips `node_modules`, `.venv`,
  `__pycache__`, `dist`, `build`, `.git`).
- Chunks each file, embeds with **`all-MiniLM-L6-v2`** (local, free, 384-dim),
  batched in groups of 100.
- Upserts into a Qdrant collection (`owner__repo`), cosine distance, `size=384`.
- **Idempotent:** skips re-indexing when the collection already has points;
  point ids are **deterministic UUIDs** (`uuid5`) so re-runs overwrite instead
  of duplicating. Original string id preserved in the payload.
- Errors are caught + logged; `index_repo` never raises; one bad file is skipped
  without killing the run.
- **Windows-portable** checkout path (`tempfile.gettempdir()/devagent`,
  override via `DEVAGENT_INDEX_ROOT`).

### Step 1.3 — `app/retrieval/retriever.py` ✅
Hybrid retrieval the agent's Explorer node will call.
- **Semantic search** — embeds the query (same `all-MiniLM-L6-v2` model) and
  runs Qdrant vector search (top-20).
- **Keyword search** — **BM25** (`rank-bm25`) over chunk text, with snake_case /
  camelCase identifier sub-tokenisation for better code recall.
- **Fusion** — **reciprocal rank fusion** (RRF, k=60); a chunk surfaced by both
  retrievers ranks above one surfaced by only one.
- **AST-aware expansion** — each fused result's `calls` are resolved to their
  defining chunks and appended (`source="expanded"`), giving the code-writer
  self-contained context.
- Errors caught + logged; returns `[]` on failure or empty index.

### Tests (13 total)
- `tests/test_chunker.py` — Python chunks, file-level fallback, callee
  extraction, JavaScript chunking.
- `tests/test_indexer.py` — file walk + skip-dirs, per-file chunking, batch
  sizing (`[100, 100, 5]`), collection params (`size=384`), re-index skip, clone
  command, deterministic UUID point ids.
- `tests/test_retriever.py` — semantic+keyword fusion ordering, dependency
  expansion, empty-corpus handling, RRF agreement, identifier tokenisation.

---

## What's next (immediate)

**Phase 1 (Retrieval) is complete.** Next is **Phase 2, Step 2.1 —
`agent/state.py`**: the typed LangGraph state schema that threads issue text,
plan, retrieved context, generated diff, and test results through the nodes.
This unblocks the Planner and Explorer nodes (the Explorer wraps
`retriever.retrieve`).

---

## Phased roadmap

### Phase 1 — Retrieval layer  ✅ complete
| Step | Module | Status |
|---|---|---|
| 1.1 | `chunker.py` | ✅ done |
| 1.2 | `indexer.py` | ✅ done |
| 1.3 | `retriever.py` | ✅ done |

### Phase 2 — Agent (LangGraph state machine)  🟡 in progress
| Step | Module | Purpose | Status |
|---|---|---|---|
| 2.1 | `agent/state.py` | Typed graph state schema | ✅ done |
| — | `agent/llm.py` | Shared OpenRouter chat client (injectable) | ✅ done |
| 2.2 | `agent/nodes/planner.py` | Decompose issue → structured subtasks | ✅ done |
| 2.3 | `agent/nodes/explorer.py` | Call the retriever, assemble context | ⏳ next |
| 2.4 | `agent/nodes/writer.py` | Generate unified diff (LLM via OpenRouter) | 🔴 |
| 2.5 | `agent/nodes/tester.py` | Run tests in sandbox, parse failures | 🔴 |
| 2.6 | `agent/nodes/pr.py` | Open the pull request | 🔴 |
| 2.7 | `agent/graph.py` | Wire nodes + self-repair loop (max 3 retries) | 🔴 |

### Phase 3 — Backend / webhook  🔴
| Step | Module | Purpose |
|---|---|---|
| 3.1 | `app/main.py` | FastAPI app, `/webhook` + `/webhook/manual` |
| 3.2 | `github/client.py` | PyGithub wrapper (read issue, push branch, open PR) |
| 3.3 | `sandbox/docker.py` | Docker SDK execution wrapper for tests |

### Phase 4 — Observability  🔴
- MLflow run logging per node; LangSmith trace integration.

### Phase 5 — Dashboard  🔴
- Next.js: run timeline, diff viewer, test output, retry history.

### Phase 6 — Evals  🔴
- `evals/benchmark.py` + curated issue set; reproduce README's benchmark table.

---

## Indicative timeline

> Estimates assume steady part-time progress from **2026-06-11**. Dates shift
> with actual pace — this is a plan, not a commitment.

| Phase | Scope | Target window |
|---|---|---|
| Phase 1 | chunker + indexer + retriever | ✅ done 2026-06-11 |
| Phase 2 | Agent nodes + graph + self-repair loop | 2026-06-19 → 2026-07-10 |
| Phase 3 | FastAPI webhook + GitHub + sandbox | 2026-07-11 → 2026-07-31 |
| Phase 4 | MLflow / LangSmith observability | 2026-08-01 → 2026-08-08 |
| Phase 5 | Next.js dashboard | 2026-08-09 → 2026-08-29 |
| Phase 6 | Benchmark + eval harness | 2026-08-30 → 2026-09-12 |

**First end-to-end run (issue → PR):** targeted at the end of Phase 3
(~2026-07-31), with everything before it being the minimum needed to close the
loop once.

---

## Known gaps / debt
- README still claims OpenAI `text-embedding-3-small`; actual build uses local
  `all-MiniLM-L6-v2` (384-dim). README to be reconciled before release.
- LLM provider is **OpenRouter** (`base_url: https://openrouter.ai/api/v1`) —
  not yet wired (no LLM calls until Phase 2).
- `.env` is referenced but not auto-loaded yet; `python-dotenv` is in
  requirements and will be loaded at the FastAPI entrypoint (Phase 3).
- TypeScript / Go language support: future (chunker is Python + JS only today).
