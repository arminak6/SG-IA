# SG-IA-WIKI

An AWS Bedrock-powered Q&A application built around the persistent LLM Wiki
pattern: immutable uploaded sources, stable manager-maintained sources, an
LLM-maintained Markdown wiki, and an explicit schema that governs ingestion and
querying.

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

## Run with Docker

The WIKI Docker stack is independent from RAG and uses different host ports:

- Streamlit UI: <http://localhost:8503>
- FastAPI documentation: <http://localhost:8002/docs>

To reuse the existing ignored `aws_credentials.json`, copy the environment
template and select that file:

```powershell
cd WIKI
Copy-Item .env.example .env
```

Then edit `.env` and set:

```text
WIKI_AWS_CREDENTIALS_FILE=./aws_credentials.json
```

Start the backend and UI:

```powershell
docker-compose up -d --build
```

Check them with:

```powershell
docker-compose ps
docker-compose logs --tail 100 wiki-api
```

The host `backend/raw/` directory is mounted read-only except for the nested
`backend/raw/manager-knowledge/` directory. A confirmed addition creates one
stable subject file there; later updates replace that file instead of adding
documents. Any legacy `manager-actions/` files are read-only under the parent
mount. `backend/wiki/` and `backend/feedback/` are mounted read/write.
These private directories are excluded from the image build and from Git.

Stop the WIKI stack without deleting the Docling model cache:

```powershell
docker-compose down
```

Docling conversion is local. The application calls Bedrock directly for model
and embedding requests and does not upload source documents to S3.

## AWS configuration

For local development, copy `aws_credentials.example.json` to
`aws_credentials.json` and fill in the values. The local credential file is
ignored by git and is never returned by the API. The backend also supports the
standard Boto3 credential chain when explicit keys are omitted.

The IAM identity needs permission to invoke the configured Bedrock model,
including `bedrock:InvokeModel` for the Converse API.

## Hybrid section search (Version 2)

Question answering combines deterministic keyword search over generated Wiki
pages with vector similarity over their Markdown sections. A section is formed
from a heading and its body. Semantic section hits are aggregated into unique
parent-page candidates before results reach the answer agent, so a page cannot
occupy several candidate slots.

The section match is only a navigation hint. The answer agent receives the
matching heading and excerpt, chooses the relevant candidates, and must read
the complete original parent Markdown page before using or citing it. The
system does not embed or retrieve chunks from `backend/raw`; the compiled Wiki
remains the query-time knowledge layer.

Amazon Titan Text Embeddings V2 is enabled by default with 512 dimensions.
Section embeddings are cached locally in
`backend/wiki/.semantic-index.json`. Each section has a stable identity and
content hash, so editing one section re-embeds only that section while unchanged
sections reuse their vectors. The Version 1 page-vector cache is rebuilt
automatically on the first query or ingestion after upgrading. No vector
database is required. If embeddings are unavailable, Q&A falls back to keyword
search instead of failing.

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
confirm whether each search used `hybrid_section`, `lexical`, or
`lexical_fallback`. `debug.retrieval_diagnostics` records each query, the final
candidate order, lexical and semantic ranks, and the best matching sections.

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

## Trusted-manager actions (POC)

The Streamlit chat keeps a random `session_id`. Normal messages are always Q&A,
including later questions in the same session. A manager action starts only
through the UI controls or an explicit command: `/fix`, `/update`, or `/add`.
The API always returns a structured preview showing the action type, whether
source knowledge changes, and whether derived Wiki maintenance will occur.
Nothing is applied until the manager chooses **Confirm action** or sends
`/confirm` (plain `approve` is also accepted); **Cancel action** or `/cancel`
discards the draft. Incomplete action details are labeled **more information
required** and receive a clarification question; they are not awaiting approval.

When an action button is selected, the normal Q&A composer is disabled. Enter
only a short human sentence in **What changed?**; the UI adds `/fix`, `/update`,
or `/add` automatically and removes a duplicate same-mode prefix. If the API
fails, the form and its text remain available for retry. Replacement prose sent
to normal chat is not applied—it receives guidance to use **Update knowledge**.
After an API restart, an orphan `/confirm` or `/cancel` explains that the draft
must be previewed again instead of entering Q&A.

- `add_knowledge` creates one stable
  `raw/manager-knowledge/<subject>.md` source, then normal ingestion creates its
  source summary and any useful canonical pages.
- `update_knowledge` atomically replaces that same stable source and rewrites
  its existing source summary and canonical pages. It cannot create another
  Wiki page, the obsolete value is removed from active knowledge, and the
  semantic index replaces the changed sections. If integration fails, the raw
  source is rolled back to its previous complete value.
- `fix_answer` creates no raw source. After confirmation, an isolated review
  must prove the manager correction from existing complete Wiki pages and pass
  the normal evidence guardrail. The application then adds verified guidance to
  one existing evidence page, preserves provenance, adds graph links to every
  supporting page, refreshes search, and stores a private regression/audit JSON
  under `backend/feedback/answer-fixes/` that is never indexed as knowledge. If
  existing evidence does not support the correction, the application changes
  no files and presents a converted `update_knowledge` proposal that requires a
  second explicit confirmation.

Conversation drafts are process-memory state and disappear when the API
restarts. Applied manager actions persist.

The explicit command determines the initial action type; the model cannot
silently switch it or ask the manager to classify it again. Cited Wiki pages
and manager sources supply the target scope automatically. Only the
post-confirmation evidence review may visibly
convert an unsupported fix into a pending update, and that transition requires
another confirmation. While a draft is pending, the manager must clarify,
confirm, or cancel it before returning to Q&A. Nothing is written until the
confirmation for the currently displayed action.

For `/update`, the text supplied after the command is retained as manager audit
input while the preview shows one complete merged current value. Still-valid
knowledge is retained, explicitly corrected facts are replaced, and an
incremental instruction is never persisted as the whole source snapshot.
Clarification is reserved for missing or ambiguous facts, not action type or
internal scope. A recurring statement such as "every year" is a complete
effective period, so it proceeds directly to preview instead of asking for a
calendar year. Before commit, update validation rejects derived Wiki rewrites
that drop critical numeric/percentage details or a stated confirmation
condition, or introduce a numeric/date claim absent from all of the page's
current raw sources. A separate semantic review also checks every material
staged claim against complete current raw sources; existing Wiki prose is not
evidence. If the manager explicitly asks for exact or verbatim wording, at
least one canonical page must preserve the complete statement. The updater
gets one bounded opportunity to repair its staged pages before rollback.
Grounded answers apply the same principle to material percentage and
confirmation qualifiers—including stated timing and communication method—
attached to the requested value.

This is intentionally a trusted-manager POC, not a production authorization
model. There is no login, role check, approval queue, rollback UI, or named
approver identity yet. Do not expose the correction-capable endpoint to
untrusted users. A production version must add authenticated RBAC, durable
audit identity, review/rollback, conflict policy, and abuse controls. For a fair
RAG-versus-WIKI comparison, approved corrections must also be added to the
shared source corpus and reindexed by RAG.

## API checks

- `GET /documents` lists pending/ingested sources.
- `POST /wiki/update` ingests selected pending sources sequentially.
  Manager-knowledge sources are deliberately skipped here so bulk ingestion
  cannot bypass confirmation and existing-page update restrictions.
- `GET /wiki/pages` exposes page summaries and raw provenance.
- `GET /wiki/lint` checks schema, provenance, links, and index coverage.
- `GET /wiki/lint` also reports incoming/outgoing graph weaknesses as warnings.
- `POST /wiki/lint/repair-links` performs bounded semantic review and adds
  validated bidirectional related-page links without permitting model-authored
  prose changes. `max_links` defaults to 12 and is capped at 50.
- `POST /chat` returns a comparison-ready answer with structured citations,
  status, usage, latency, model ID, an optional 0-10 evidence-confidence score,
  and wiki navigation debug data. Supplying a stable `session_id` enables the
  trusted-manager action preview/confirmation flow and adds an optional
  structured `manager_action` object. `correction` remains as a compatibility
  alias during the POC.
- Read-only Q&A retries the complete answer-agent operation exactly once when a
  Bedrock request fails. Successful first attempts have no extra model cost;
  recovered responses expose `debug.answer_attempts: 2` and
  `debug.answer_retry_applied: true`. A second Bedrock failure still returns a
  sanitized 503, and non-Bedrock application errors are never hidden by retries.
  Within an operation, an invalid structured answer followed by free text can
  receive one fresh bounded submission reminder instead of failing immediately.

## Answer confidence

After the answer agent submits a grounded response, an isolated Bedrock verifier
checks the response against the complete cited Wiki pages. The final 0-10 score
combines claim support (45%), question coverage (20%), lexical/semantic retrieval
agreement (15%), source consistency (10%), and evidence quality/provenance (10%).
An unsupported material claim caps an answered response at 5, and an unexplained
source conflict subtracts 2 points.

For an `insufficient_knowledge` response, the score instead measures confidence
that abstaining was appropriate. The response text makes that target clear. The
API returns the number as `confidence_score`, and Streamlit displays only
`Confidence score: X.X/10` directly below the answer.

For the current POC, the verifier is advisory by default: its score and stable
warning reasons remain visible, but it does not replace the answer. This keeps
the model's behavior observable during experiments. Set
`LLM_WIKI_ANSWER_GUARDRAIL_ENABLED=true` to restore fail-closed enforcement. In
that mode, a factual answer becomes a citation-free `insufficient_knowledge`
response when it contains an unsupported material claim, has claim support
below 0.8, covers less than 0.7 of the question, has evidence quality below 0.6,
leaves a source conflict unexplained, or verification is unavailable.
`debug.guardrail` reports whether enforcement was enabled/applied, the original
status, verification availability, and warning reasons.

## Tests

The test suite uses a scripted Bedrock client, so it does not call AWS or spend
model tokens:

```powershell
python -m unittest discover -s backend/tests -v
python -m unittest discover -s frontend/tests -v
```
