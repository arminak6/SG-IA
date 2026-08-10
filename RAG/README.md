# SG-IA standard RAG

This directory contains the API-first standard RAG side of the SG-IA
RAG-versus-LLM-Wiki comparison. It is independently runnable and does not
import code from `WIKI/`.

The current milestone implements document ingestion and semantic retrieval:

- a FastAPI backend accepts files and reports asynchronous ingestion status;
- Docling extracts PDF, DOCX, and PPTX locally while preserving ordered
  elements, headings, tables, pages, and bounding boxes;
- Markdown, text, CSV, and JSON are read locally as UTF-8 sources;
- structure-aware chunks are embedded with Amazon Bedrock Titan Text
  Embeddings V2 and stored in Qdrant using cosine similarity;
- a Streamlit client uploads documents, monitors jobs, lists indexed sources,
  and displays retrieved evidence with scores and provenance;
- identical file content is reused instead of indexed twice.

Answer generation is deliberately the next milestone. The future `/chat`
endpoint will combine retrieved evidence with a Bedrock generation model and
return a comparison-ready answer/citation/timing envelope. Until then, the
Streamlit Chat tab explains the boundary and the Retrieval tab exposes what the
generator would receive.

## Architecture

```text
Streamlit UI --HTTP--> FastAPI backend
                         |-- local Docling extraction
                         |-- structure-aware chunking
                         |-- Bedrock Titan embeddings
                         `-- Qdrant cosine retrieval
```

Streamlit contains no ingestion or retrieval logic. This keeps the backend
usable by the eventual side-by-side RAG/WIKI interface.

## Run everything with Docker

Prerequisites are Docker, Docker Compose, AWS credentials that can invoke the
configured Bedrock embedding model, and outbound access to the selected AWS
region. This pipeline calls Bedrock directly. It does **not** upload documents
to S3 or create any other AWS storage resource.

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
| `POST` | `/documents` | multipart upload and ingestion queue |
| `GET` | `/ingestions/{job_id}` | extraction/indexing progress or safe error |
| `GET` | `/documents` | indexed document manifests |
| `GET` | `/documents/{document_id}` | one indexed manifest |
| `POST` | `/search` | semantic chunks, scores, citations, and latency |

Example retrieval request:

```json
{
  "query": "What procedure applies?",
  "top_k": 5,
  "document_ids": null
}
```

The backend saves a document manifest only after the Qdrant count exactly
matches the number of generated chunks. If ingestion fails, it removes that
document's Qdrant points and exposes a sanitized failure through the job API.

## Initial technical decisions

- Qdrant is the vector database.
- Titan Text Embeddings V2 defaults to 512 dimensions and normalized vectors.
- Qdrant uses cosine distance.
- chunks default to approximately 600 tokens with 60-token overlap only when
  an oversized indivisible element must be split;
- tables, figures, formulas, and code remain atomic where possible;
- chunk size, overlap, embedding model/dimension, collection, upload limit,
  OCR, ports, and region are configurable through environment variables.

These are corpus-agnostic starting values, not tuned rules for the current
benchmark. Any later quality tuning must be evaluated on held-out documents
and paraphrased questions as required by the repository policy.

## Privacy and Git

`RAG/data/`, Qdrant storage, model caches, `.env`, credentials, and generated
artifacts are excluded from Git. Uploaded source bytes and extracted content
remain local (or in Docker named volumes); only embedding requests are sent
directly to Amazon Bedrock.

