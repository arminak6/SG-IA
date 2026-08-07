# SG-IA Project Knowledge Base

This file is the durable project memory for coding agents. Read it first at the
start of every session. Do not scan the whole repository unless the current
task requires it; inspect only the relevant files after reading this file.

## Project goal

Build and compare two document-grounded Q&A chatbot approaches over the same
source material:

1. `RAG/`: a standard retrieval-augmented generation (RAG) implementation.
2. `WIKI/`: a knowledge-wiki implementation based on the LLM Wiki pattern or
   Google's Open Knowledge Format (OKF). The final choice, or whether both will
   be evaluated, is not yet decided.

The end product must let a user submit one question and see the RAG and WIKI
answers simultaneously for direct comparison.

## Repository responsibilities

- `material/` is the common source corpus. Both approaches must use the same
  in-scope documents so their results can be compared fairly.
- `RAG/backend/` owns ingestion, indexing/retrieval, generation, and an HTTP
  API for the standard RAG approach.
- `RAG/frontend/` owns a Streamlit UI used to develop and test RAG independently.
- `WIKI/backend/` owns wiki ingestion/maintenance, grounded Q&A, and an HTTP
  API for the WIKI approach.
- `WIKI/frontend/` owns a Streamlit UI used to develop and test WIKI independently.
- A final comparison interface will call both backend APIs for the same
  question and display both results together. Its location and framework are
  still to be decided.

Some of these paths may not exist yet. Create them when implementing that
component; do not treat their absence as a change in the intended architecture.

## Non-negotiable design constraints

- Backend functionality is the primary product surface and must always be
  available through APIs. Streamlit must be an API client, not the place where
  ingestion, retrieval, wiki reasoning, or answer-generation logic lives.
- Keep RAG and WIKI independently runnable and testable.
- Preserve separate Streamlit frontends under both `RAG/` and `WIKI/` during
  development, even after a combined comparison UI is introduced.
- Use identical source scope and, where practical, the same question set and
  evaluation conditions for both approaches.
- Answers should expose evidence/citations and useful debug metadata so quality
  and behavior can be compared rather than judged only by prose.
- Keep approach-specific internals separate. Put only genuinely shared API
  contracts, evaluation fixtures, or utilities in shared code.
- Never commit credentials, API keys, tokens, or local secret files.
- Treat the current documents, entities, filenames, benchmark questions,
  ground-truth answers, and scores only as temporary evaluation examples. They
  are not product requirements and must never be embedded in prompts or code.
- Never improve benchmark scores by hardcoding question IDs, wording, expected
  answers, document-specific mappings, filenames, thresholds, or special-case
  response rules. Improvements must be corpus-agnostic and suitable for unseen
  production documents and questions.
- Validate quality changes on held-out documents and paraphrased/unseen
  questions, not only on the current benchmark, to detect leakage and overfit.

## Intended end-to-end flow

1. Select an in-scope subset of `material/`.
2. Ingest that same subset independently into RAG and WIKI.
3. Validate each backend through its API and its own Streamlit frontend.
4. Send the same question to both backend APIs.
5. Show both answers, citations/evidence, timings, and relevant diagnostics
   side by side.
6. Evaluate the approaches with a shared test question set and agreed metrics.

## API comparison direction

When API contracts are added or revised, keep the two Q&A responses easy to
normalize. At minimum, plan for these conceptual fields:

- answer text
- source citations/provenance
- approach identifier (`rag` or `wiki`)
- latency/timing data
- optional debug metadata (for example retrieved chunks for RAG or wiki pages
  used for WIKI)
- clear errors and insufficient-evidence responses

Exact endpoint paths and schemas are not yet frozen. Record the decision here
when they are agreed.

## Current state (2026-08-07)

- The project root is initialized as a `main`-branch Git monorepo containing
  the sibling `RAG/` and `WIKI/` implementations. The root README presents the
  project as a fair RAG-versus-LLM-Wiki comparison.
- The root `.gitignore` excludes credentials, private source documents,
  generated Wiki knowledge, the private benchmark fixture, evaluation results,
  reports, local indexes/databases/models, caches, and editor state. Placeholder
  documentation keeps the intended data directories visible in a fresh clone.
- `material/` contains the shared candidate corpus in several formats.
- `RAG/` exists but is currently empty.
- `WIKI/` already contains a FastAPI backend, Streamlit frontend, tests, and a
  persistent Markdown LLM Wiki proof of concept powered by AWS Bedrock.
- `WIKI/backend/AGENTS.md` is the operating contract specifically for wiki
  ingestion and Q&A. It remains authoritative inside `WIKI/backend/` and must
  be read before changing that component.
- The WIKI proof of concept was reset to a clean baseline on 2026-08-06: its
  previous fictitious test source and generated pages were removed. Alignment
  with the shared `material/` corpus remains future work.
- WIKI now uses local Docling conversion for PDF, DOCX, and PPTX. All 24 files
  currently staged in `WIKI/backend/raw/` passed an extraction-only audit and
  are ingested into 61 wiki pages; none are pending.
- The WIKI API now returns a comparison-ready Q&A envelope with approach,
  answer status, structured wiki/raw citations, model usage, latency, model ID,
  and evidence-navigation debug data. The future RAG API should normalize to
  this conceptual contract.
- WIKI ingestion state is content-hash based, multi-page ingestion commits roll
  back on failure, and `GET /wiki/lint` validates page schema, provenance,
  local links, and index coverage.
- WIKI graph lint now reports weak incoming/outgoing page relationships, and
  `POST /wiki/lint/repair-links` uses bounded semantic review to add only
  backend-authored, validated, bidirectional cross-links.
- WIKI Q&A now uses hybrid retrieval over generated Wiki pages: deterministic
  lexical search plus configurable Amazon Bedrock Titan Text Embeddings V2
  similarity, fused by rank. Page embeddings are content-hash cached locally,
  refreshed after ingestion, and safely fall back to lexical search during an
  embedding outage. Raw-document chunks are never placed in this index.
- `test_QA/mateial/ground_truth_qa.json` is the initial shared evaluation fixture: 25
  Italian questions grounded in the WIKI raw corpus, with required answer
  points, source locators, evidence, and two insufficient-knowledge controls.
  `test_QA/mateial/README.md` documents the provisional comparison metrics.
- Automated semantic evaluation will use a separately configurable,
  high-capability Amazon Bedrock model as the LLM judge. Where practical, the
  judge model should differ from the chatbot model, use deterministic settings,
  and return structured point-level correctness and groundedness judgments.
- `test_QA/WIKI/evaluate_wiki.py` implements the WIKI benchmark harness. It
  records complete API responses, invokes the independent Bedrock judge,
  checks required points and consulted Wiki evidence, classifies failures, and
  emits JSONL, CSV, summary, and reproducibility-manifest artifacts. Its tests
  use fakes and make no AWS calls. The live evaluator retries transient WIKI
  API failures up to three times and malformed judge responses once.
- The clean final WIKI benchmark run is
  `test_QA/WIKI/results/20260807T085528Z-3050df`: all 25 chatbot and judge
  calls completed using Claude Opus 5 as the independent EU Bedrock judge.
  Headline results are 3.84/5 average correctness, 69.5% required-point
  coverage, 86.4% groundedness, and 16/23 answerable cases scoring at least 4.
  The management report is available as editable DOCX and presentation-ready
  PDF under `test_QA/WIKI/report/`; a two-page chart-focused executive PDF is
  available at `output/pdf/SG-IA_WIKI_Two_Page_Executive_Summary.pdf`.
- The post-hybrid-search comparison run is
  `test_QA/WIKI/results/20260807-hybrid-consolidated`. It contains one
  successful chatbot and Claude Opus 5 judgment for each of the same 25 cases;
  its manifest records bounded retry recovery for four transient failures.
  Compared with the clean baseline, average correctness moved from 3.84 to
  3.92, score >=4 from 72% to 76%, groundedness from 86.4% to 92.3%, expected
  source recall from 91.3% to 100%, and average server latency from 6.03 to
  6.64 seconds. This is a single stochastic comparison, not a statistical
  conclusion. The standalone two-page report for the hybrid run is
  `output/pdf/SG-IA_WIKI_Two_Page_Executive_Summary_Hybrid.pdf`; detailed
  Italian and English question/answer records are `README.md` and
  `README_EN.md` inside the consolidated result directory.

## Decisions still required

Do not silently choose these during unrelated work. Resolve them when a task
depends on them, then update this file:

- LLM Wiki vs Google OKF vs evaluating both for the WIKI approach.
- Exact documents under `material/` that form the initial comparison corpus.
- Models, embedding model, vector database, and chunking strategy for RAG.
- Model/provider and knowledge-generation pipeline for WIKI.
- Common API request/response schema.
- Final scoring weights and any expansion of the initial `test_QA/` benchmark.
- Location/technology of the final side-by-side comparison interface.

## How to maintain this memory

Update this file whenever the user establishes or changes a durable goal,
constraint, architecture decision, major dependency, or milestone. Keep it
concise and factual. Move an item from "Decisions still required" into an
appropriate section once decided, and update the dated current-state section
after meaningful implementation changes.

Do not use this file as a task log or paste large code details into it. Detailed
component instructions belong in a nearer `AGENTS.md`; implementation history
belongs in version control.
