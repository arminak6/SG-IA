# WIKI Component Knowledge Base

Read this file first for any work under `WIKI/`. It is the durable development
memory for this component. After reading it, inspect only files relevant to the
current task instead of rescanning the entire component.

The repository-level project purpose and comparison constraints are documented
in `../AGENTS.md` and remain authoritative. The more specific
`backend/AGENTS.md` is the runtime operating contract for LLM-managed wiki
ingestion and Q&A; read it before changing backend wiki behavior.

## Component purpose

`WIKI/` implements the knowledge-wiki side of the SG-IA comparison. It must
turn the same source material used by `RAG/` into persistent, auditable
knowledge pages and answer grounded questions through an API. Its results will
eventually be displayed beside RAG results for the same question.

The current proof of concept follows the LLM Wiki pattern. Whether the final
comparison uses this implementation, Google's Open Knowledge Format (OKF), or
evaluates both remains undecided; do not silently treat that choice as final.

## Current architecture

- `backend/main.py`: FastAPI entry point.
- `backend/app/`: configuration, AWS Bedrock integration, wiki agent,
  repository boundary, and application service.
- `backend/app/extraction.py`: lazy local Docling conversion for PDF, DOCX,
  and PPTX sources.
- `backend/raw/`: immutable source documents available for ingestion.
- `backend/wiki/`: generated Markdown knowledge base.
- `backend/wiki/index.md`: application-maintained knowledge catalog.
- `backend/wiki/log.md`: append-only application operation history.
- `backend/AGENTS.md`: schema and behavior supplied to the wiki-maintaining LLM.
- `backend/tests/`: backend tests using a scripted Bedrock client.
- `frontend/`: Streamlit API client and its tests.

Keep domain logic in the FastAPI/backend layer. Streamlit is for independent
testing and visualization and must call the API rather than duplicate backend
logic.

## Current state and progress

### 2026-08-06: clean baseline

- Removed the previous `chatEg.txt` test source about the imaginary Nexora
  Technologies company.
- Removed every generated source, concept, and entity page derived from that
  source.
- Reset `backend/wiki/index.md` to an empty catalog and
  `backend/wiki/log.md` to an empty operation history.
- The WIKI knowledge store now contains no test knowledge and is ready for the
  agreed SG-IA material corpus.
- Existing backend, frontend, and automated tests were preserved.

### 2026-08-06: structured document extraction

- Selected and verified Docling 2.118 as the WIKI document-conversion layer.
- Added local PDF, DOCX, and PPTX ingestion while keeping original raw files
  immutable and avoiding persistent converted copies.
- PDF extraction uses OCR/layout/table processing in CPU/eager mode for
  predictable Windows operation without Visual C++ build tools.
- Added PDF page and PPTX slide markers to extracted Markdown. DOCX keeps its
  heading/document flow because Word pages are not stable semantic boundaries.
- Raised the raw-file limit from 2 MB to 25 MB. Conversion is bounded to 500
  pages and 600,000 extracted characters.
- Verified real extraction of all 24 current raw documents: 24 passed and zero
  failed. The full CPU audit took about 11.5 minutes; the slowest PDF took about
  175 seconds. No documents were ingested into the wiki during the audit.
- Streamlit now lists only supported sources and requires explicit selection
  before ingestion, avoiding an accidental all-corpus Bedrock run.
- Automated verification now includes 28 backend tests and 13 frontend tests.

### 2026-08-06: real ingestion progress

- The raw corpus contains 24 supported documents.
- All 24 documents are successfully ingested and represented by exact source
  provenance across 61 generated wiki knowledge pages. No documents remain
  pending.
- Earlier failed attempts included `00.Assumptions.docx` and
  `01.Introduzione_GruppoImprese.docx` exceeding the agent tool-step limit;
  both are now successfully ingested. `02.Storia Gruppo Tonetto e Londei
  (GTL).docx` ended without reading its raw
  source; `03.Organigramma.pdf`, `1 Manuali SGIA/GESTIONE DOCUMENTI(1) (1).pptx`,
  and `1 Manuali SGIA/GESTIONE UTENTI.pdf` received Bedrock
  `ValidationException` responses.
- Subsequent attempts ingested all three emergency-plan floor-plan PDFs.
  `03.Organigramma.docx` exceeded the tool-step limit, the AI policy ended
  without reading its raw source, and the safety/health instructions received
  a Bedrock `ValidationException`.
- Streamlit pending cards now show an explicit **Ingest this document** button,
  in addition to the multi-document selector at the top of the sidebar.
- Fixed disabled pending-card ingestion buttons caused by the UI's 1.5-second
  health-check timeout. Repository status now reads each wiki page once instead
  of once per source/page combination, `/health` counts pages without building
  full metadata, and the UI allows 15 seconds for a health response. With the
  current corpus the live health request dropped from about 9 seconds to 0.15
  seconds; all pending-document buttons render enabled when Bedrock is
  configured.

### 2026-08-06: bounded ingestion workflow correction

- Removed the contradictory instruction that told the LLM to update
  `wiki/index.md`; knowledge-page generation is unchanged, while the application
  remains the sole deterministic owner of index rebuilding.
- Focused ingestion on the mandatory source-summary page and only useful
  concept/entity/synthesis updates, raised the safety ceiling from 16 to 24
  rounds, and added safe recent-tool-name diagnostics when the ceiling is hit.
- Retried `01.Introduzione_GruppoImprese.docx` successfully in about 43 seconds.
  It has exact provenance in nine committed pages, all are present in the
  rebuilt index, and no partial-write behavior was introduced.
- A later `03.Organigramma.docx` retry exposed two distinct budget failures:
  one run was still writing at round 24, while another spent all rounds in
  discovery and produced no staged work. The agent now uses enforced raw-read,
  bounded-discovery, and write-only phases and repairs premature completion.
- At the round boundary, valid staged work commits only when it includes a
  source-summary page and passes schema, exact-provenance, and prior-citation
  preservation checks; otherwise nothing is written. Under this controller,
  `03.Organigramma.docx` ingested successfully in about 43 seconds with exact
  provenance in eight indexed pages.
- The AI policy PDF then exposed that a model can repeatedly return free text
  instead of invoking even the sole mandatory raw-read tool. Raw extraction is
  now performed deterministically by the backend before the first model call;
  the exact path and extracted content are supplied as explicitly untrusted
  source data, making "ended without reading" structurally impossible.
- Retried the AI policy successfully in about 54 seconds. It has exact
  provenance in one source-summary and seven concept pages, all indexed. The
  final source, `Sicurezza lavoratori/04.Istruzioni di sicurezza e salute.pdf`,
  initially received a Bedrock `ValidationException`; its extracted input was
  verified as small and valid Unicode. A later diagnostic attempt encountered
  a transient endpoint connection failure, and the next normal API retry
  succeeded, confirming the provider failure was transient rather than a
  document or request-shape defect.
- The final safety-and-health source has exact provenance in its source-summary
  and two concept pages. A full integrity audit found all 24 sources ingested,
  all 61 knowledge pages indexed, and zero pages without raw-source provenance.
- Bedrock requests now retry exactly once after a `ValidationException`. The
  retry is safe because validation rejection occurs before any model output or
  wiki tool action; a repeated rejection still fails normally without looping.

### 2026-08-06: backend integrity and comparison readiness

- Replaced path-reference-only ingestion status with a SHA-256 manifest. A raw
  document is ingested only while its current bytes match the committed digest
  and its recorded provenance pages still exist. All 24 existing sources were
  migrated and remain correctly reported as ingested.
- Knowledge-page validation now parses YAML and enforces title, page type,
  ISO date, existing normalized raw sources, directory/type agreement, exact
  `## Sources` provenance, and valid local wiki/raw links.
- Page writes, index rebuilding, and ingestion-manifest updates now form a
  rollback transaction. A process-wide file lock serializes writers across API
  workers; operation-log failure no longer converts committed knowledge into a
  reported ingestion failure.
- Added deterministic `GET /wiki/lint`; the current 61-page wiki passes with
  zero schema, provenance, link, or index errors. Repaired eight bad wiki links,
  three bad raw-document links, two nonstandard provenance headings, and one
  escaped title/index label found by the audit.
- Q&A cannot cite a page until a prior tool round has returned that page, and a
  citation must expose all raw provenance carried by that page. `/chat` now
  returns `approach`, evidence status, structured citations, usage, latency,
  model ID, and pages/searches used for later RAG comparison.
- Automated verification now includes 35 backend tests and 13 frontend tests;
  all pass without live AWS calls.

### 2026-08-06: semantic graph maintenance

- `GET /wiki/lint` now reports knowledge-graph metrics and warnings for pages
  without incoming links, outgoing links, or either. Structural validity remains
  separate, so warnings do not incorrectly make a sound wiki invalid.
- Added bounded `POST /wiki/lint/repair-links`. Bedrock must read both complete
  pages and give a semantic reason before proposing a relationship; at least one
  endpoint must have a weak graph position.
- The model cannot edit page prose in this workflow. The repository adds only
  validated, bidirectional `Related pages` links, updates dates, validates the
  resulting pages, and commits them transactionally.
- The original graph baseline was 61 directed links, 34 pages without incoming
  knowledge links, 37 without outgoing links, and 26 isolated when `index.md`
  is excluded. A live bounded repair run raised the graph to 78 links and
  reduced those figures to 26, 28, and 22 respectively.
- Automated verification now includes 38 backend tests and 13 frontend tests;
  all pass without live AWS calls.

### 2026-08-06: initial comparison benchmark

- Added `../test_QA/mateial/ground_truth_qa.json` with 25 Italian evaluation cases
  grounded directly in 18 raw documents: 23 answerable cases and two
  insufficient-knowledge controls.
- Each answerable case records required facts, exact raw paths, document
  locators, and short evidence. The shared benchmark is intended for both WIKI
  and the future RAG backend; it is not generated from wiki-page prose.

### 2026-08-07: automated WIKI evaluation harness

- Added `../test_QA/WIKI/evaluate_wiki.py`, which runs the shared questions
  through `/chat` and uses a separately configurable, high-capability Bedrock
  model for structured semantic judging.
- Evaluation covers required-point correctness, 1–5 scoring, groundedness
  against pages actually read, source citations, abstention, failure diagnosis,
  latency, usage, optional cost, and grouped summaries. Run artifacts include
  dataset, corpus-manifest, and judge-prompt hashes for reproducibility.
- The evaluator's offline tests use fake API and Bedrock responses and make no
  paid calls. Live read-only preflight verified a valid 61-page Wiki.
- Added bounded retries for transient `/chat` failures and malformed judge
  output. The retry-enabled final run
  `../test_QA/WIKI/results/20260807T085528Z-3050df` completed 25/25 chatbot and
  25/25 judge calls with Anthropic Claude Opus 5 as the independent EU Bedrock
  judge. It measured 3.84/5 average correctness, 69.5% required-point coverage,
  86.4% groundedness, and 16/23 answerable cases scoring at least 4.
- Added an executive evaluation report in PDF and editable DOCX form under
  `../test_QA/WIKI/report/`. Its deployment conclusion is controlled internal
  pilot with human review, not unmonitored production use.

### 2026-08-10: hybrid section search Version 2 benchmark

- Upgraded hybrid retrieval from one embedding per Wiki page to content-hash
  cached Markdown heading sections. Lexical and semantic ranks are fused and
  section hits are aggregated into unique parent pages; the answer agent still
  reads complete parent pages before answering or citing them.
- The local 61-page Wiki produced 186 semantic sections. The live benchmark
  prewarmed this Version 2 cache before latency measurement.
- The consolidated 25-case run is
  `../test_QA/WIKI/results/20260810-hybrid-section-v2-consolidated`; two primary
  API failures were replaced only by their first successful recovery results.
  Claude Opus 5 measured 3.88/5 correctness, 73.7% required-point coverage,
  89.5% groundedness, and 91.3% expected-source recall. Detailed Italian and
  English records are in the run directory and the standalone two-page report
  is `../output/pdf/SG-IA_WIKI_Two_Page_Executive_Summary_Hybrid_Section_V2.pdf`.

### 2026-08-10: independent Docker UI stack

- `compose.yaml` now runs the WIKI FastAPI backend and Streamlit API client
  independently on host ports 8002 and 8503, so it can run beside RAG.
- The existing local `backend/raw/` corpus is mounted read-only and
  `backend/wiki/` is mounted read/write; neither private directory is copied
  into the image. The ignored local Bedrock JSON can be selected through
  `WIKI_AWS_CREDENTIALS_FILE`.
- Docling uses a persistent Docker model-cache volume. The stack passed local
  API/UI health checks while the RAG stack remained available on its ports.

### 2026-08-14: explicit manager-action routing

- Normal chat messages now always use Q&A, including new questions after a
  previous answer. Manager maintenance begins only through `/fix`, `/update`,
  or `/add`, and the selected action type is enforced deterministically.
- The Streamlit UI provides separate manager-action controls plus explicit
  preview, clarification, confirmation, and cancellation states. Confidence on
  an insufficient-knowledge response is labeled as abstention confidence.
- Action mode now disables the normal Q&A composer, accepts a short human
  sentence, automatically adds/deduplicates the action prefix, and preserves
  form text after API failure. Unmarked replacement prose returns manager-form
  guidance rather than a Q&A error; confirmation without an in-memory draft
  returns a specific restart/re-preview message.
- Ready drafts require one confirmation. Plain `approve`/`approva` are exact
  confirmation aliases; incomplete drafts are instead labeled as requiring
  more information and remain non-writable until clarification is complete.
- Generic regression coverage validates short Update/Fix/Add inputs and uses a
  temporary repository for Add cleanup. A live current-corpus smoke test
  confirmed that a short manager update changed the stable source and derived
  page, after which the same question returned the new cited value.

### 2026-08-17: manager-update merge and fidelity safeguards

- The selected Update/Fix/Add control is authoritative. The structuring model
  cannot reclassify it or ask the manager to choose the action again, and cited
  Wiki pages provide internal update scope automatically.
- Updates retain the manager's exact sentence as audit input but preview and
  persist one complete merged current value. Incremental wording no longer
  replaces the entire stable manager source.
- Clarification is limited to genuinely missing or ambiguous factual details.
  An independent merge review removes unsupported inferred wording before the
  preview. Pre-commit deterministic and semantic checks reject derived Wiki
  rewrites that lose approved
  numeric/percentage details or confirmation conditions, or introduce numeric
  dates or other material claims absent from all current page sources. An
  explicit exact/verbatim request must appear intact in a canonical page. The
  agent receives a bounded repair turn before fail-closed rollback. Grounded
  answers must retain percentage and confirmation qualifiers, including stated
  timing and communication method, attached to the
  requested value, and an invalid structured submission can receive one fresh
  bounded submit reminder.
- The affected Sinergia mid-summer source and its two existing Wiki pages were
  repaired through the confirmed API workflow. They now preserve the approved
  statement that the meeting is held on 13 July with 100% certainty and that
  Sinergia will email everyone one week beforehand to confirm the date.
- Manager additions now materialize the approved value verbatim into one stable
  source-summary page and, when the path is free, one canonical subject page;
  no generation pass can embellish trusted new knowledge. Updates derive their
  complete writable scope from manifest ownership, preserve stable subject,
  scope, and effective-period metadata, and may atomically delete a wholly
  obsolete page only when the manager source is its sole provenance.
- Answer output strips model-written Sources sections and rejects calculated or
  unsupported clock times, AM/PM, and timezone/local-time qualifiers. Bedrock
  transport now uses a 60-second read timeout with one SDK attempt. Two live
  temporary knowledge lifecycles passed add/answer plus three successive
  update/answer rounds without page-count growth; all temporary data was removed.

### 2026-08-26: private user preferences and session chat context

- The Streamlit POC now requires a simple claimed user name and keeps private
  per-user data under `backend/user_data/`, outside `backend/wiki/` and its
  search/indexing flow. Each user has structured `preferences.json`, a readable
  `profile.md`, and exact per-session JSONL transcripts; all are Git-ignored
  and mounted persistently in both Docker Compose entry points.
- Grounded Q&A receives the active user's preferences and a bounded window of
  only the current session's earlier turns. The prompt treats them as
  presentation/conversation context, never factual evidence; normal Wiki reads
  and citations remain mandatory. Calls without a user remain stateless for
  benchmark compatibility.
- Explicit durable preference statements and `/remember` are saved, while a
  profile editor supports direct review and changes. **New chat** and **Sign
  out** delete only the active session transcript and manager draft state while
  retaining the profile and preferences. This remains an unauthenticated local
  POC and must not be treated as an identity or privacy boundary.
- Read-only Q&A now retries the complete answer operation once when the answer
  model fails to invoke the required structured submission tool after its
  existing reminder. This addresses observed intermittent 503 responses while
  keeping unrelated validation failures and all write workflows non-retryable.

## Data rules

- Use the same agreed subset of `../material/` as the RAG implementation.
- Do not ingest candidate material until its source scope has been agreed.
- Raw inputs are immutable once placed under `backend/raw/`; generated
  knowledge belongs only under `backend/wiki/`.
- Binary extraction is performed in memory. Do not add committed converted
  Markdown copies or extraction caches unless a measured need justifies a
  deliberate cache design.
- Generated claims must retain exact source provenance and expose uncertainty
  or missing evidence instead of guessing.
- Never mix demonstration/fictitious content with the real comparison corpus.

## Development and housekeeping rules

- Keep this component independently runnable and testable through its API and
  its own Streamlit frontend.
- Preserve API responses suitable for later normalization with RAG: answer,
  citations/provenance, approach identity, timing, useful debug metadata, and
  clear insufficient-evidence/error states.
- Do not commit credentials or expose `aws_credentials.json`.
- Keep the tree clean. Temporary scripts, converted files, debug dumps,
  screenshots, plots, and one-off artifacts created only to complete a task
  must be removed after their final output is produced, unless they are useful
  reusable project assets or the user asks to retain them.
- Do not remove or overwrite unrelated user files or changes.

## Verification commands

From `WIKI/`:

```powershell
python -m unittest discover -s backend/tests -v
python -m unittest discover -s frontend/tests -v
```

Tests use a scripted Bedrock client and should not make paid AWS calls.

## Decisions still open

- LLM Wiki versus Google OKF versus evaluating both.
- Initial in-scope files from `../material/` and how they are staged into
  `backend/raw/` without creating divergent RAG/WIKI corpora.
- Production model/provider configuration.
- Final cross-backend evaluation metrics; the WIKI response envelope is now the
  starting contract for normalization with RAG.
- Whether XLSX should be added later; PDF, DOCX, and PPTX are now supported
  through Docling.

## Maintaining this memory

Update the current-state section after meaningful milestones and record durable
component decisions when the user makes them. Keep this file concise: it is a
project map and decision record, not a transcript or detailed task log.
