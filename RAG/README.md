# SG-IA standard RAG

This directory contains the API-first standard RAG side of the SG-IA
RAG-versus-LLM-Wiki comparison. It is independently runnable and does not
import code from `WIKI/`.

The current implementation provides document ingestion, semantic retrieval,
and grounded question answering:

- a FastAPI backend accepts files and reports asynchronous ingestion status;
- Docling extracts PDF, DOCX, and PPTX locally while preserving ordered
  elements, headings, tables, pages, and bounding boxes;
- Markdown, text, CSV, and JSON are read locally as UTF-8 sources;
- structure-aware chunks are embedded with Amazon Bedrock Titan Text
  Embeddings V2 and stored in Qdrant using cosine similarity;
- a Bedrock generation model answers only from retrieved evidence and submits
  the exact chunk IDs used as citations;
- unsupported questions return `insufficient_evidence` without citations;
- a Streamlit client uploads documents, monitors jobs, lists indexed sources,
  displays retrieval diagnostics, and calls the grounded `/chat` API;
- identical file content is reused instead of indexed twice.

## Architecture

```text
Streamlit UI --HTTP--> FastAPI backend
                         |-- local Docling extraction
                         |-- structure-aware chunking
                         |-- Bedrock Titan embeddings
                         |-- Qdrant cosine retrieval
                         `-- grounded Bedrock answer generation
```

Streamlit contains no ingestion or retrieval logic. This keeps the backend
usable by the eventual side-by-side RAG/WIKI interface.

## Run everything with Docker

Prerequisites are Docker, Docker Compose, AWS credentials that can invoke the
configured Bedrock embedding and generation models, and outbound access to the
selected AWS region. This pipeline calls Bedrock directly. It does **not**
upload documents to S3 or create any other AWS storage resource.

1. Copy `RAG/.env.example` to `RAG/.env` and add temporary AWS credentials, or
   export them in the shell before starting Compose. Never commit `.env`.
2. From this directory run:

   ```powershell
   docker-compose up -d --build
   ```

3. Open:

   - Streamlit: <http://localhost:8502>
   - FastAPI documentation: <http://localhost:8001/docs>
   - Qdrant dashboard: <http://localhost:6337/dashboard>

Useful checks:

```powershell
docker-compose ps
docker-compose logs --tail 100 rag-api
```

Stop the stack without deleting indexed data:

```powershell
docker-compose down
```

Adding `-v` deletes the named Qdrant, RAG-data, and Docling-cache volumes, so
use it only when a full local reset is intended.

## Run without Docker

Start Qdrant separately on port 6333, then use two terminals:

```powershell
cd RAG\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
$env:QDRANT_URL="http://127.0.0.1:6333"
uvicorn main:app --reload --port 8001
```

```powershell
cd RAG\frontend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:RAG_API_URL="http://127.0.0.1:8001"
streamlit run app.py --server.port 8502
```

The normal boto3 credential chain is used outside Docker, so an existing AWS
profile, environment variables, or an IAM role can authorize Bedrock. A local
JSON credentials file can also be selected with `RAG_AWS_CREDENTIALS_FILE`;
files matching `*credentials*.json` are ignored by Git.

## API contract

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | API/Qdrant readiness |
| `GET` | `/models` | effective non-secret model configuration |
| `POST` | `/documents` | multipart upload and ingestion queue |
| `GET` | `/ingestions/{job_id}` | extraction/indexing progress or safe error |
| `GET` | `/documents` | indexed document manifests |
| `GET` | `/documents/{document_id}` | one indexed manifest |
| `POST` | `/search` | semantic chunks, scores, citations, and latency |
| `POST` | `/chat` | grounded answer, citations, usage, timings, and retrieval debug data |

Example retrieval request:

```json
{
  "query": "What procedure applies?",
  "top_k": 5,
  "document_ids": null
}
```

Example grounded question request:

```json
{
  "question": "Come si inserisce un nuovo utente in SGIA?",
  "session_id": null,
  "top_k": 10,
  "document_ids": null
}
```

`/chat` embeds the question, retrieves 24 semantic candidates from Qdrant,
reranks them, expands adjacent chunks inside the same document section, and
passes the best 8–10 evidence chunks to the answer model. Before generation, a
deterministic corpus-agnostic coverage check compares requested question facets
with the evidence. When coverage is incomplete, retrieval is retried once with
the missing facets before generation. The answer model then requires a structured
`submit_grounded_answer` result. The backend rejects unknown citation IDs and
returns only the chunks selected by the model. The response uses the
comparison-ready `approach`, `status`, `answer`, `citations`, `usage`,
`latency_ms`, `timings`, `model_id`, and `debug` fields.

The backend saves a document manifest only after the Qdrant count exactly
matches the number of generated chunks. If ingestion fails, it removes that
document's Qdrant points and exposes a sanitized failure through the job API.

## Initial technical decisions

- Qdrant is the vector database.
- Titan Text Embeddings V2 defaults to 512 dimensions and normalized vectors.
- Qdrant uses cosine distance.
- the initial answer model is `openai.gpt-oss-20b-1:0`, matching the current
  WIKI answer model so retrieval architecture can be compared more fairly;
- generation uses Bedrock Converse tool use, temperature 0.1, and a validated
  evidence-ID submission rather than trusting free-form citation text;
- chunks default to approximately 600 tokens with 100-token overlap only when
  an oversized indivisible element must be split;
- RAG v1.2 retrieves 24 initial semantic candidates, combines semantic and
  lexical evidence coverage during reranking, expands same-section neighbours,
  and sends 10 final chunks by default;
- a pre-generation coverage check can make one bounded retrieval retry when
  numeric or lexical question facets are missing from the final context;
- tables, figures, formulas, and code remain atomic where possible;
- chunk size, overlap, embedding model/dimension, collection, upload limit,
  OCR, ports, and region are configurable through the model registry and/or
  environment variables.

## Model configuration

The tracked [`config/models.json`](config/models.json) is the central,
non-secret inventory and default configuration for extraction, embedding, and
generation models. It records:

- Docling plus its layout, RapidOCR, and TableFormer components;
- the Bedrock embedding model and vector dimensions;
- the Bedrock generation model, API, temperature, and output limit.

Deployment environment variables override JSON values when present. This keeps
container-specific configuration possible without editing the tracked file.
Compose mounts this file read-only into the backend container, so generation
model or parameter changes take effect after `docker-compose restart rag-api`;
an image rebuild is not required. Changing the embedding model or dimensions
still requires a new compatible Qdrant collection name and re-ingestion.
Docling-managed component identifiers
are inventory metadata; changing their implementations may also require a
compatible Docling dependency or extractor-code update.

These are corpus-agnostic starting values, not tuned rules for the current
benchmark. Any later quality tuning must be evaluated on held-out documents
and paraphrased questions as required by the repository policy.

## Shared 25-question benchmark

`test_QA/RAG/evaluate_rag.py` runs the same Italian fixture and independent
Claude Opus 5 judging method used for WIKI. It records complete API responses,
cited RAG chunks, point-level correctness, groundedness, expected-source
recall, latency, token usage, and a corpus/model reproducibility snapshot.

The 13 August 2026 baseline run is
`test_QA/RAG/results/20260813T095001Z-3e053d`. Twenty-four of 25 API calls were
successfully answered and judged; one structured-answer validation failure was
reproducible after bounded retries. Across the 24 judged cases, average
correctness was 4.38/5, required-point coverage 87.2%, groundedness 94.8%, and
expected-source recall 81.8%; 22/24 scored at least 4. Both insufficient-
knowledge controls abstained correctly. Generated results and reports remain
ignored by Git.

The v1.2 run is `test_QA/RAG/results/20260813T115206Z-4a98ba`. All 25 API and
judge calls completed. Average correctness was 4.52/5, required-point coverage
90.4%, groundedness 96.2%, expected-source recall 89.1%, and 23/25 cases scored
at least 4. Both insufficient-knowledge controls abstained correctly and there
were no false abstentions. This is one stochastic run, not a statistical
conclusion. Its report is
`output/pdf/SG-IA_RAG_Two_Page_Executive_Summary_V1_2.pdf`.

## Privacy and Git

`RAG/material/`, `RAG/data/`, Qdrant storage, model caches, `.env`, credentials,
and generated artifacts are excluded from Git. Uploaded source bytes and
extracted content remain local or in Docker named volumes. Embedding inputs and
the bounded chunks retrieved for each question are sent directly to the
configured Amazon Bedrock models.
