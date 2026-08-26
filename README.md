# SG-IA: RAG vs LLM Wiki

SG-IA is an experimental comparison between two strategies for answering
questions over the same private document collection:

1. **RAG (Retrieval-Augmented Generation)** retrieves relevant document chunks
   for each question and gives those chunks to a language model.
2. **LLM Wiki** first transforms the source material into a persistent,
   linked knowledge wiki and answers questions by navigating that wiki.

The goal is to evaluate both approaches fairly using the same source scope,
questions, ground-truth answers, models where practical, and evaluation
conditions. The final interface will show both answers, citations, timings,
and diagnostics side by side.

## Current status

| Component | Status | Purpose |
| --- | --- | --- |
| `WIKI/` | Working proof of concept | FastAPI backend, persistent Markdown LLM Wiki, hybrid section search with full-page reading, Streamlit UI, ingestion, citations, linting, and tests |
| `RAG/` | RAG v1.2 working | FastAPI ingestion, Docling extraction, Bedrock embeddings and generation, Qdrant retrieval/reranking, grounded `/chat`, Docker Compose, and an API-only Streamlit client |
| `test_QA/` | Evaluation harness available | Shared semantic evaluation design and WIKI benchmark runner |
| `comperision/` | Working Streamlit client | Concurrently send one question to both backends and compare answers, evidence, timings, and diagnostics side by side |
| `deployment/` | Portable deployment tools | Unified startup, shared-corpus bootstrap and validation, plus safe state export/import |

## Repository layout

```text
SG-IA/
|-- material/       Local shared source corpus; private files are not committed
|-- RAG/            Retrieval-augmented generation implementation
|-- WIKI/           LLM Wiki implementation
|-- comperision/    Side-by-side Streamlit API client
|-- deployment/     Portable startup, corpus, validation, and backup tools
|-- compose.yaml    Unified six-service Docker Compose stack
|-- test_QA/        Shared evaluation code and local benchmark fixtures
|-- output/         Generated reports; not committed
`-- AGENTS.md       Durable architecture and development decisions
```

## Fair comparison principles

- Use the same in-scope source documents for RAG and LLM Wiki.
- Ask both systems the same questions.
- Compare semantic correctness against human-authored ground truth.
- Record citations, groundedness, latency, token usage, and failures.
- Keep the backends independently runnable and expose both through HTTP APIs.
- Do not apply retrieval-ranking metrics to LLM Wiki unless it exposes an
  explicit ranked retrieval stage.

## Data and security

This repository intentionally excludes all local or generated data that may
contain company information:

- AWS credentials, `.env` files, and Streamlit secrets;
- documents under `material/` and `WIKI/backend/raw/`;
- generated Wiki pages under `WIKI/backend/wiki/`;
- RAG uploads, Docling artifacts, job manifests, and Qdrant data;
- the private ground-truth fixture and completed chatbot evaluations;
- generated reports, local databases, indexes, model files, and caches.

Only example configuration files may be committed. Never put real keys or
internal documents in Git, even temporarily. A fresh clone therefore contains
the application code but not the private corpus or generated knowledge base.

## Complete application quick start

The recommended deployment on another computer is the root Compose stack. It
starts Qdrant, both APIs, all three Streamlit interfaces, and connects the
comparison client directly to both backends:

```powershell
Copy-Item .env.example .env
Copy-Item WIKI/aws_credentials.example.json WIKI/aws_credentials.json
# Add temporary AWS credentials and point SGIA_AWS_CREDENTIALS_FILE to that file.
.\deployment\start.ps1
```

Open the comparison UI at <http://localhost:8504>. RAG remains available at
<http://localhost:8502> and LLM Wiki at <http://localhost:8503>.
When the private corpus manifest is present, startup idempotently ingests any
missing manifest documents into both systems and verifies exact alignment.

A folder copy includes the code and bind-mounted WIKI files, but Docker named
volumes are stored outside the folder. Use the shared-manifest bootstrap to
rebuild both knowledge bases, or the verified export/import workflow to carry
the existing RAG index. See [`deployment/README.md`](deployment/README.md) for
the exact fresh-computer procedure and safety guarantees.

## LLM Wiki quick start

See [`WIKI/README.md`](WIKI/README.md) for complete installation and API/UI
instructions. In brief:

```powershell
cd WIKI
Copy-Item aws_credentials.example.json aws_credentials.json
python -m pip install -r backend/requirements.txt
python -m pip install -r frontend/requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd WIKI
streamlit run frontend/app.py
```

Place approved local source documents in `WIKI/backend/raw/`, ingest them from
the UI, and keep them outside Git.

Alternatively, run the independent Docker stack:

```powershell
cd WIKI
docker-compose up -d --build
```

The Docker WIKI UI is <http://localhost:8503> and its API documentation is
<http://localhost:8002/docs>. See [`WIKI/README.md`](WIKI/README.md) for the
ignored credential-file setting and bind-mounted data behavior.

## RAG quick start

The first RAG milestone supports document upload, asynchronous Docling
ingestion, structure-aware chunks, Bedrock Titan embeddings, Qdrant retrieval,
and retrieval inspection in Streamlit. From `RAG/`, copy `.env.example` to
`.env`, configure temporary Bedrock credentials, and run:

```powershell
docker-compose up -d --build
```

Open Streamlit at <http://localhost:8502> and FastAPI documentation at
<http://localhost:8001/docs>. See [`RAG/README.md`](RAG/README.md) for API
details and privacy notes.

## Independent side-by-side client

For component development, after independently starting RAG and WIKI with the
same source scope, the comparison client can still be run separately:

```powershell
cd comperision
docker-compose up -d --build
```

Open <http://localhost:8504>. One question is submitted concurrently to both
`/chat` APIs; LLM Wiki is shown on the left and RAG on the right. See
[`comperision/README.md`](comperision/README.md) for configuration and local
development instructions.

## Evaluation

The evaluator code is documented in [`test_QA/WIKI/README.md`](test_QA/WIKI/README.md).
The real ground-truth fixture and run outputs are local-only because they are
derived from the private corpus. Offline evaluator tests remain safe to run:

```powershell
python -m unittest discover -s test_QA/WIKI/tests -v
```

