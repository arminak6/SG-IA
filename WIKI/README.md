# SG-IA-WIKI

An AWS Bedrock-powered Q&A application built around the persistent LLM Wiki
pattern: immutable raw sources, an LLM-maintained Markdown wiki, and an explicit
schema that governs ingestion and querying.

## Layout

```text
backend/raw/          Immutable source documents
backend/wiki/         Bedrock-maintained Markdown knowledge base
backend/AGENTS.md     Wiki schema and agent workflow
backend/main.py       FastAPI application
frontend/app.py       Streamlit interface
aws_credentials.json Local Bedrock configuration (git-ignored)
```

## Install

```powershell
python -m pip install -r backend/requirements.txt
python -m pip install -r frontend/requirements.txt
```

## Run

Start the API from the project root:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, start Streamlit:

```powershell
streamlit run frontend/app.py
```

The frontend uses `http://127.0.0.1:8000` by default. Override it with the
`LLM_WIKI_API_URL` environment variable when needed.

## AWS configuration

For local development, copy `aws_credentials.example.json` to
`aws_credentials.json` and fill in the values. The local credential file is
ignored by git and is never returned by the API. The backend also supports the
standard Boto3 credential chain when explicit keys are omitted.

The IAM identity needs permission to invoke the configured Bedrock model,
including `bedrock:InvokeModel` for the Converse API.

## Hybrid Wiki search

Question answering combines the existing deterministic keyword search with
vector similarity over the generated Wiki pages. It does not embed or retrieve
chunks directly from `backend/raw`, so the compiled Wiki remains the knowledge
layer used at query time.

Amazon Titan Text Embeddings V2 is enabled by default with 512 dimensions. Wiki
page embeddings are cached in `backend/wiki/.semantic-index.json` and keyed by
the page content hash. New or changed pages are embedded after ingestion;
unchanged pages reuse their cached vector. An existing Wiki builds its cache on
the first semantic query. If embeddings are unavailable, Q&A automatically
falls back to keyword search instead of failing.

The local JSON configuration supports:

```json
{
  "embedding_model_id": "amazon.titan-embed-text-v2:0",
  "embedding_dimensions": 512,
  "semantic_search_enabled": true
}
```

Environment overrides are `BEDROCK_EMBEDDING_MODEL_ID`,
`BEDROCK_EMBEDDING_DIMENSIONS`, and
`LLM_WIKI_SEMANTIC_SEARCH_ENABLED`. Set the last value to `false` to use only
keyword search. Chat responses report `debug.search_modes` so evaluation can
confirm whether each search used `hybrid`, `lexical`, or `lexical_fallback`.

## Supported sources

The ingestion layer supports UTF-8 Markdown, text, JSON, and CSV directly. PDF,
DOCX, and PPTX files are converted locally to structured Markdown with Docling
when the wiki agent reads them. Original files under `backend/raw` remain
unchanged, nested relative paths remain part of wiki provenance, and extracted
PDF pages or PowerPoint slides receive source markers for grounded summaries.

Docling runs locally and does not send document content to a parsing service.
PDF conversion uses OCR, layout analysis, and table extraction in CPU/eager mode
so Windows does not require Visual Studio C++ build tools. Office documents are
usually quick; image-heavy PDFs can take several minutes. Process one selected
document at a time through the Streamlit interface when monitoring progress.

Raw files are limited to 25 MB, converted documents to 500 pages, and extracted
content to 600,000 characters. A conversion that is empty, unsupported, or over
one of these limits fails clearly without creating partial wiki knowledge.

Successful ingestion records a SHA-256 digest in the application-owned wiki
manifest. Editing a raw file therefore makes it pending again. Knowledge-page,
index, and manifest changes roll back together if a commit fails.

## API checks

- `GET /documents` lists pending/ingested sources.
- `POST /wiki/update` ingests selected pending sources sequentially.
- `GET /wiki/pages` exposes page summaries and raw provenance.
- `GET /wiki/lint` checks schema, provenance, links, and index coverage.
- `GET /wiki/lint` also reports incoming/outgoing graph weaknesses as warnings.
- `POST /wiki/lint/repair-links` performs bounded semantic review and adds
  validated bidirectional related-page links without permitting model-authored
  prose changes. `max_links` defaults to 12 and is capped at 50.
- `POST /chat` returns a comparison-ready answer with structured citations,
  status, usage, latency, model ID, and wiki navigation debug data.

## Tests

The test suite uses a scripted Bedrock client, so it does not call AWS or spend
model tokens:

```powershell
python -m unittest discover -s backend/tests -v
python -m unittest discover -s frontend/tests -v
```
