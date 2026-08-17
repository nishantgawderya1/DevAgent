<div align="center">

# 🤖 DevAgent

### Autonomous GitHub Issue Resolver

*Assign an issue. Get a pull request.*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C3C3C?style=flat)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Sandboxed-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Architecture](#architecture) · [Tech Stack](#tech-stack) · [Quickstart](#quickstart) · [Supported Languages](#supported-languages)

</div>

---

## What is DevAgent?

DevAgent is a fully autonomous software engineering agent that resolves GitHub issues end-to-end — without human intervention. You open an issue. DevAgent reads it, navigates your codebase, writes a fix, runs your existing tests, and opens a pull request.

It is **not** a copilot. It does not autocomplete your code while you type. It is an agent that takes a task and completes it — the same way a junior engineer would, except it works at 3am and never asks for clarification on `good first issue` bugs.

```
Issue opened → Agent plans → Codebase explored → Patch written → Tests pass → PR opened
                                                                      ↑
                                                              Self-repair loop
                                                         (retries on test failure)
```

---

## Why DevAgent?

Existing tools solve adjacent problems:

| Tool | What it does | What's missing |
|---|---|---|
| GitHub Copilot | Autocomplete while you type | Requires a human in the loop |
| Devin | Full autonomous dev agent | Closed source, $500/month, black box |
| SWE-agent | Research prototype | No self-hosted webhook integration, no live dashboard |
| AutoCodeRover | AST-guided repair | No end-to-end GitHub workflow |

DevAgent fills the gap: **open-source, self-hostable, production-integrated** via GitHub webhooks, with a clean observability dashboard and a formalized self-repair loop.

---

## Architecture

DevAgent is built as a **LangGraph state machine** with five specialized nodes. Each node has a single responsibility; the graph handles routing, retries, and state passing.

```
GitHub Webhook
      │
      ▼
┌─────────────┐
│   Planner   │  Decomposes issue into structured subtasks (JSON)
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Codebase Explorer│  AST-aware retrieval via tree-sitter + Qdrant
└──────┬───────────┘
       │
       ▼
┌─────────────┐
│ Code Writer │  Generates unified diff, applies to Docker sandbox
└──────┬──────┘
       │
       ▼
┌─────────────┐       ┌──────────────┐
│ Test Runner │──fail─▶ Code Writer  │  Self-repair loop (max 3 retries)
└──────┬──────┘       └──────────────┘
       │ pass
       ▼
┌─────────────┐
│  PR Creator │  Opens pull request via GitHub API
└─────────────┘
```

### What makes it different: AST-aware retrieval

Most code RAG systems chunk files by character count. This destroys context — a function gets split mid-body, imports get separated from their usage sites, class methods lose their parent.

DevAgent uses **tree-sitter** to parse every file into its actual AST structure before indexing. Chunks are at the function and class level, with full metadata: file path, line range, direct imports, and caller/callee relationships. When the retriever fetches a relevant function, it automatically expands context to include its dependencies.

This gives the code writer agent meaningful, self-contained context — not arbitrary text windows.

### Self-repair loop

When generated tests fail, DevAgent doesn't give up. The test runner node parses structured output (pytest/jest), identifies which assertions failed, and feeds the failure reason back into the code writer as a new prompt. This loop runs up to 3 times before the run is marked as failed and logged for review.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM calls | LangChain (OpenAI / Anthropic / Gemini) |
| Code parsing | tree-sitter (Python, JavaScript) |
| Vector store | Qdrant (local Docker) |
| Code embeddings | `all-MiniLM-L6-v2` (local, 384-dim) |
| Sandboxed execution | Docker SDK |
| Backend API | FastAPI |
| GitHub integration | PyGithub + Webhooks |
| Observability | MLflow + LangSmith |
| Dashboard | Next.js |
| Infra | Docker Compose |

---

## Quickstart

### Prerequisites
- Docker + Docker Compose
- Python 3.11+
- Node.js 18+
- GitHub App (for webhook + PR creation)
- OpenAI / Anthropic / Gemini API key

### 1. Clone and install

```bash
git clone https://github.com/nishantgawderya1/devagent
cd devagent
pip install -r requirements.txt
cd dashboard && npm install
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in your `.env`:

```env
# LLM
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...

# GitHub App
GITHUB_APP_ID=...
GITHUB_PRIVATE_KEY_PATH=./certs/github-app.pem
GITHUB_WEBHOOK_SECRET=...

# Qdrant (local)
QDRANT_HOST=localhost
QDRANT_PORT=6333

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
```

### 3. Start services

```bash
docker-compose up -d   # Qdrant + MLflow
uvicorn app.main:app --reload --port 8000
cd dashboard && npm run dev
```

### 4. Connect your GitHub repo

Install the DevAgent GitHub App on your repository. On every new issue, DevAgent's webhook fires automatically.

Or trigger manually:

```bash
curl -X POST http://localhost:8000/webhook/manual \
  -H "Content-Type: application/json" \
  -d '{"repo": "owner/repo", "issue_number": 42}'
```

---

## Project Structure

```
devagent/
├── app/
│   ├── main.py              # FastAPI app + webhook handler
│   ├── agent/
│   │   ├── graph.py         # LangGraph state machine definition
│   │   ├── nodes/
│   │   │   ├── planner.py   # Issue decomposition
│   │   │   ├── explorer.py  # AST retrieval
│   │   │   ├── writer.py    # Patch generation
│   │   │   ├── tester.py    # Sandboxed test execution
│   │   │   └── pr.py        # GitHub PR creation
│   │   └── state.py         # LangGraph state schema
│   ├── retrieval/
│   │   ├── chunker.py       # tree-sitter AST chunker
│   │   ├── indexer.py       # Qdrant indexing pipeline
│   │   └── retriever.py     # Hybrid semantic + BM25 retrieval
│   ├── sandbox/
│   │   └── docker.py        # Docker SDK execution wrapper
│   └── github/
│       └── client.py        # PyGithub wrapper
├── dashboard/               # Next.js frontend
├── evals/
│   ├── benchmark.py         # Evaluation runner
│   └── issues/              # Curated benchmark issue set
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Dashboard

The Next.js dashboard gives you full visibility into every agent run:

- **Run timeline** — see each node's execution time and status
- **Diff viewer** — side-by-side view of the generated patch
- **Test output** — full pytest/jest output per run
- **Retry history** — see what changed between attempts
- **MLflow integration** — every run logged with full metadata

---

## Supported Languages

| Language | AST parsing | Test runner |
|---|---|---|
| Python | ✅ tree-sitter | pytest |
| JavaScript | ✅ tree-sitter | jest |
| TypeScript | 🔜 Coming soon | jest |
| Go | 🔜 Planned | go test |

---

## Limitations

DevAgent v1 is deliberately scoped. It works best on:
- **Bug fixes** in existing codebases (not greenfield features)
- **Repos under 500 files** (larger repos can be indexed but retrieval quality degrades)
- **Python and JavaScript** only (tree-sitter parsers for other languages coming)
- **Issues with clear reproduction steps** in the issue body

It will struggle with:
- Architectural changes spanning many files
- Issues requiring external API knowledge not in the codebase
- Repos without any existing test coverage (test runner node has nothing to validate against)

These are known limitations and active areas of improvement.

---

## Roadmap

- [ ] TypeScript + Go language support
- [ ] Multi-file patch generation (currently single-file only)
- [ ] Slack/Discord notification integration
- [ ] Fine-tuned code writer model on patch dataset
- [ ] SWE-bench full evaluation
- [ ] Cloud-hosted version (waitlist)

---

## Contributing

Contributions welcome. Please read `CONTRIBUTING.md` before opening a PR.

Areas where help is most needed:
- Additional tree-sitter language parsers
- Benchmark issue curation (more languages, more repos)
- Dashboard UI improvements
- Test coverage for agent nodes

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

Built by [Nishant Gawderya](https://nishantgawderya.me) · [LinkedIn](https://linkedin.com/in/nishantgawderya) · [Twitter](https://twitter.com/nishantgawderya)

*If DevAgent saves you time, give it a ⭐*

</div>