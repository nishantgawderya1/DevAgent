# DevAgent — Work Update

_Last updated: 2026-08-18 (Phases 1–3 complete — webhook, sandbox and GitHub backends are in; the loop closes)_

A running log of what's built, what's next, and the phased plan to get from the
current retrieval layer to the full autonomous issue-resolver described in the
README.

---

## TL;DR

DevAgent is being built bottom-up, and the core loop is now **structurally
complete**: a webhook receives an issue, the graph plans it, retrieves context,
writes a patch, runs the suite in a container, repairs on failure, and opens a
PR. What remains is *observation and evidence* — dashboard, tracing, benchmark.

| Layer | Status |
|---|---|
| Retrieval (chunk → index → retrieve) | ✅ Complete (3 of 3 modules) |
| Agent nodes (LangGraph) | ✅ Complete (5 nodes + graph) |
| Backend / webhook (FastAPI) | ✅ Complete |
| GitHub integration | ✅ Complete |
| Sandbox (Docker exec) | ✅ Complete |
| Dashboard (Next.js) | 🔴 Not started |
| Evals / benchmark | 🔴 Not started |

**Tests:** 87 passing (`pytest -q`).

> **Runnable today?** The app boots and every route is live. What has *not*
> happened yet is a real run: no live LLM call, no container actually executing
> a suite, no PR opened against a real repository. Every backend is tested
> against fakes and against real local git, but the first true end-to-end run
> still needs credentials and a running Docker daemon. That is the immediate
> next task, and it is where the interesting bugs will be.

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

### Phase 2 — `app/agent/` ✅
The five nodes from the README, wired into a compiled LangGraph state machine.

- **`state.py`** — typed `AgentState` TypedDict, `should_retry` (the loop
  condition), `record_failure`. Deliberately imports no LangGraph.
- **`llm.py`** — shared OpenRouter chat client, lazily constructed, injectable.
- **`nodes/planner.py`** — issue → ordered JSON subtasks.
- **`nodes/explorer.py`** — folds the plan's subtasks into the retrieval query
  (they name the symbols BM25 needs), calls `retriever.retrieve`, and flattens
  hits to plain dicts so the state stays checkpointable.
- **`nodes/writer.py`** — plan + context → unified diff. Unwraps markdown fences
  the model adds despite being told not to, extracts target files from the diff,
  and on re-entry folds the failing test output back into the prompt. Owns
  `retry_count`, incremented only when re-entered from a failure.
- **`nodes/tester.py`** — orchestration and parsing only; execution is an
  injected `runner`. Parses pytest and jest failures, truncates output keeping
  the tail, and decides the branch: pass → PR, fail under budget → writer,
  fail over budget → failed.
- **`nodes/pr.py`** — composes branch (`devagent/issue-N`), title, and a body
  that shows the plan, files changed, and repair count. GitHub calls injected.
- **`graph.py`** — wires the nodes; every edge gated on `status != "failed"` so
  any node can end the run, with the tester as the only real branch.

**Design note:** the tester and PR nodes require their backend to be injected
rather than defaulting to something real. Model-generated patches should not
execute unsandboxed by default, and that constraint is what keeps the whole
graph runnable offline in tests.

### Phase 3 — backends + webhook ✅
- **`sandbox/docker.py`** — implements `TestRunner`. Clones the indexed checkout
  into a throwaway workspace (so a patch can never corrupt the tree the vectors
  were built from), applies the diff, and runs the suite in a container with
  capabilities dropped, `no-new-privileges`, and memory/CPU/PID caps.
  Auto-detects pytest vs jest from `package.json`.
  **A rejected patch is returned as a failing test run, not an exception** — so
  a malformed diff is repaired by the same loop that repairs a wrong one, and
  git's own error text is what the writer sees.
  Network is on by default because installing dependencies needs it;
  `SandboxConfig(allow_network=False)` tightens the box.
- **`github/client.py`** — implements `GitHubClient`, plus the issue read the
  webhook needs. Branch construction goes through **git** (clone, apply, commit,
  push) because a unified diff is exactly what `git apply` consumes; the PR
  itself is opened through **PyGithub**. Accepts a token or GitHub App
  credentials.
- **`app/main.py`** — FastAPI: `/webhook` (HMAC-verified, `issues.opened`),
  `/webhook/manual`, `/runs`, `/runs/{id}`, `/health`. Runs execute as
  background tasks since GitHub wants a response in seconds. Each run indexes
  the repo first — idempotent, so the cost is per-repository, not per-issue.
  **Signature verification fails closed:** an unset `GITHUB_WEBHOOK_SECRET`
  rejects every delivery rather than accepting unauthenticated ones.
- **`fsutil.py`** — `remove_tree`, shared by both backends. `shutil.rmtree`
  fails on the read-only files git creates under `.git/objects` on Windows, and
  `ignore_errors=True` hides that rather than fixing it, so every run leaked its
  workspace. Found by a cleanup assertion in the sandbox tests.

### Tests (57 total)
- `tests/test_chunker.py` — Python chunks, file-level fallback, callee
  extraction, JavaScript chunking, decorator capture (regression).
- `tests/test_indexer.py` — file walk + skip-dirs, per-file chunking, batch
  sizing (`[100, 100, 5]`), collection params (`size=384`), re-index skip, clone
  command, deterministic UUID point ids.
- `tests/test_retriever.py` — semantic+keyword fusion ordering, dependency
  expansion, empty-corpus handling, RRF agreement, identifier tokenisation.
- `tests/test_state.py` — defaults, custom retry budget, `should_retry` at each
  boundary, `record_failure`.
- `tests/test_planner.py` — object and bare-list parsing, JSON response format,
  empty plan, invalid JSON.
- `tests/test_explorer.py` — context flattening, query composition (issue +
  plan), empty retrieval, retriever errors.
- `tests/test_writer.py` — diff and target-file extraction, fence stripping,
  prompt contents, failure feedback on repair, retry accounting, multi-file diffs.
- `tests/test_tester.py` — pass/fail/budget-exhausted routing, pytest and jest
  parsing, output truncation, missing runner, runner errors.
- `tests/test_pr.py` — branch/title composition, PR body contents, repair note,
  missing client, API errors.
- `tests/test_graph.py` — end-to-end issue → PR, self-repair loop, budget
  exhaustion, short-circuit on planner and retrieval failure.
- `tests/test_sandbox.py` — patch application against **real git repos**,
  actionable rejection messages, isolation limits, workspace cleanup, indexed
  checkout left untouched, suite auto-detection, docker-unavailable handling.
- `tests/test_github_client.py` — issue read, and a branch **pushed to a real
  local bare repo** then verified to contain the patched file and not the patch
  file, default-branch targeting, auth resolution, PyGithub name-shadowing.
- `tests/test_main.py` — signature verification (valid, wrong, missing, unset),
  event and action filtering, manual trigger, payload validation, run registry.

---

## What's next (immediate)

**Phases 1–3 are complete and the app boots** — `/health`, `/runs`,
`/runs/{id}`, `/webhook`, `/webhook/manual` all respond. The next task is not
more code, it is **the first real run**:

1. Put credentials in `.env` (`OPENROUTER_API_KEY`, `GITHUB_TOKEN`,
   `GITHUB_WEBHOOK_SECRET`), start Docker Desktop, `docker-compose up -d`.
2. Point `/webhook/manual` at a small throwaway repo with a real failing test
   and watch a run go through end to end.
3. Expect the writer prompt to need iteration — it has never seen live model
   output, and diff formatting is the likeliest thing to break first.

Only then does Phase 4 (MLflow/LangSmith) earn its place, because tracing is
most useful once there is a real run to trace.

---

## Phased roadmap

### Phase 1 — Retrieval layer  ✅ complete
| Step | Module | Status |
|---|---|---|
| 1.1 | `chunker.py` | ✅ done |
| 1.2 | `indexer.py` | ✅ done |
| 1.3 | `retriever.py` | ✅ done |

### Phase 2 — Agent (LangGraph state machine)  ✅ complete
| Step | Module | Purpose | Status |
|---|---|---|---|
| 2.1 | `agent/state.py` | Typed graph state schema | ✅ done |
| — | `agent/llm.py` | Shared OpenRouter chat client (injectable) | ✅ done |
| 2.2 | `agent/nodes/planner.py` | Decompose issue → structured subtasks | ✅ done |
| 2.3 | `agent/nodes/explorer.py` | Call the retriever, assemble context | ✅ done |
| 2.4 | `agent/nodes/writer.py` | Generate unified diff (LLM via OpenRouter) | ✅ done |
| 2.5 | `agent/nodes/tester.py` | Run tests via injected runner, parse failures | ✅ done |
| 2.6 | `agent/nodes/pr.py` | Open the pull request | ✅ done |
| 2.7 | `agent/graph.py` | Wire nodes + self-repair loop (max 3 retries) | ✅ done |

### Phase 3 — Backend / webhook  ✅ complete
| Step | Module | Purpose | Status |
|---|---|---|---|
| 3.1 | `app/main.py` | FastAPI app, `/webhook` + `/webhook/manual` | ✅ done |
| 3.2 | `github/client.py` | git push + PyGithub PR creation | ✅ done |
| 3.3 | `sandbox/docker.py` | Docker SDK execution wrapper for tests | ✅ done |
| — | `app/fsutil.py` | Cross-platform tree removal | ✅ done |

### Phase 4 — Observability  🔴
- MLflow run logging per node; LangSmith trace integration.

### Phase 5 — Dashboard  🔴
- Next.js: run timeline, diff viewer, test output, retry history.

### Phase 6 — Evals  🔴
- `evals/benchmark.py` + curated issue set. **No benchmark numbers exist yet** —
  the README's results table was removed on 2026-08-18 precisely because it
  described results that had never been measured. Numbers go back only once this
  phase actually produces them.

---

## Indicative timeline

> Estimates assume steady part-time progress. Dates shift with actual pace —
> this is a plan, not a commitment.

| Phase | Scope | Target window |
|---|---|---|
| Phase 1 | chunker + indexer + retriever | ✅ done 2026-06-11 |
| Phase 2 | Agent nodes + graph + self-repair loop | ✅ done 2026-08-18 |
| Phase 3 | FastAPI webhook + GitHub + sandbox | ✅ done 2026-08-18 |
| Phase 4 | MLflow / LangSmith observability | after the first real run |
| Phase 5 | Next.js dashboard | after Phase 4 |
| Phase 6 | Benchmark + eval harness | last |

**First unattended end-to-end run (issue → PR):** every piece is now in place
and unblocked — what it needs is credentials, a running Docker daemon, and a
target repository, not more code.

---

## Known gaps / debt
- **No real LLM call has ever run.** Every node is tested against fakes, so the
  prompts in `planner.py` and `writer.py` are unvalidated against a live model.
  The writer's "unified diff only" instruction is the likeliest thing to need
  iteration — `_extract_diff` already assumes the model ignores it and emits
  markdown fences.
- **No container has actually executed a suite.** The sandbox is tested against
  a fake Docker client; the Docker daemon was not running during development, so
  image pull, dependency install, and real exit codes are unverified.
- **No PR has been opened against real GitHub.** The git half is verified for
  real (a branch is pushed to a local bare repo and inspected in the tests), but
  the PyGithub half has only ever seen fakes.
- **Installing dependencies needs network, so the sandbox allows it by default.**
  Container isolation still applies — dropped capabilities, no-new-privileges,
  memory/PID caps — but a patched test suite can reach the network unless
  `allow_network=False`. Worth revisiting if this ever runs against untrusted
  repositories.
- **Run state is in memory.** A restart loses history. Fine until the dashboard
  needs it, and contained behind `get_run` / `list_runs`.
- `.env` is referenced but not auto-loaded yet; `python-dotenv` is in
  requirements and will be loaded at the FastAPI entrypoint (Phase 3).
- **Single-file patches in practice.** `writer.py` collects multiple target files
  from a diff, but nothing has exercised a genuine multi-file patch end-to-end.
- Retrieval is untuned: `RRF_K=60` and the top-20/top-20 candidate widths are
  literature defaults, never measured against this corpus. Worth revisiting once
  Phase 6 gives a way to measure.
- TypeScript / Go language support: future (chunker is Python + JS only today).
- README reconciled 2026-08-18 — embedding model corrected, and the demo,
  benchmark, and paper sections removed as unsubstantiated.
