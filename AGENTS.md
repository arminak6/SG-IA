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
- `comperision/` owns the Streamlit comparison interface. It calls both backend
  APIs concurrently for the same question and displays both results together.
- `deployment/` owns portable orchestration, exact-corpus bootstrap and
  validation, and credential-free state export/import. Root `compose.yaml`
  starts the complete application; component Compose files remain supported.

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

## Current state (2026-08-25)

- The project root is initialized as a `main`-branch Git monorepo containing
  the sibling `RAG/` and `WIKI/` implementations. The root README presents the
  project as a fair RAG-versus-LLM-Wiki comparison.
- The root `.gitignore` excludes credentials, private source documents,
  generated Wiki knowledge, the private benchmark fixture, evaluation results,
  reports, local indexes/databases/models, caches, and editor state. Placeholder
  documentation keeps the intended data directories visible in a fresh clone.
- Root `compose.yaml` is the unified local deployment: one command starts
  Qdrant, both FastAPI backends, both independent Streamlit UIs, and the
  side-by-side comparison UI on one internal network. It preserves the existing
  RAG named-volume identities and keeps model/retrieval settings configurable
  through a documented root `.env.example`.
- `deployment/` provides a private SHA-256 corpus manifest, repeatable bootstrap
  of the exact same source bytes into both approaches, readiness/corpus
  validation, and verified RAG-volume plus WIKI-state export/import. Backups
  exclude credentials, and import refuses unsafe archives or any non-empty or
  conflicting destination. Direct Python and Streamlit dependencies are pinned
  to the versions validated in the current containers. Normal root startup
  performs the idempotent dual bootstrap and final alignment validation;
  `-SkipManifest` is reserved for intentionally empty deployments.
- `material/` contains the shared candidate corpus in several formats.
- `RAG/` now has an API-first ingestion, retrieval, and grounded-answer flow. A FastAPI
  backend accepts PDF, DOCX, PPTX, Markdown, text, CSV, and JSON uploads;
  performs local ordered Docling extraction for binary office documents;
  creates configurable structure-aware chunks; embeds them with configurable
  Amazon Bedrock Titan Text Embeddings V2 (512 dimensions by default); and
  verifies their storage in a Qdrant cosine collection before publishing an
  indexed document manifest. Content-hash deduplication prevents identical
  uploads from biasing retrieval.
- `RAG/frontend/` is a strict HTTP client. Its simplified Streamlit UI focuses
  on one document-upload area and grounded chat, with compact ingestion status
  and citations/diagnostics in optional expanders. `POST /chat` embeds each
  question, retrieves Qdrant chunks, invokes the configured Bedrock generation
  model, requires a validated structured evidence-ID submission, and returns a
  comparison-ready answer/status/citation/usage/timing/debug envelope.
  Answers follow the question's language rather than the evidence language;
  unsupported questions return `insufficient_evidence` without citations.
- RAG answers with retrieved evidence receive an advisory 0-10 confidence score
  from a separate temperature-0 Bedrock verification pass. It evaluates claim
  support, question coverage, source consistency, and evidence quality against
  the actual retrieved chunks. Verifier failure leaves the grounded answer
  intact with a null score and `verification_unavailable` debug metadata. The
  RAG UI and comparison UI display the score, while component diagnostics and
  verifier latency remain available for evaluation.
- `RAG/config/models.json` is the tracked non-secret model registry for Docling
  extraction components, Titan embeddings, and Bedrock answer generation;
  environment variables remain deployment overrides. RAG initially uses
  `openai.gpt-oss-20b-1:0` for answers, matching the current WIKI answer model,
  and Titan Text Embeddings V2 at 512 dimensions.
- The local RAG Compose deployment currently retains the agreed 24-document
  KnowledgeBase comparison subset as 496 verified Qdrant chunks. On 2026-08-13
  all backend tests passed and live `/chat` checks produced a cited grounded
  answer for an in-scope Italian question and `insufficient_evidence` for an
  out-of-scope question.
- The initial shared RAG benchmark is
  `test_QA/RAG/results/20260813T095001Z-3e053d`. It attempted the same 25 cases
  used for WIKI with GPT-OSS 20B generation and independent Claude Opus 5
  judging. Twenty-four answers completed and were judged; one structured-answer
  validation failure persisted after bounded retries. On the 24 judged cases,
  correctness was 4.38/5, required-point coverage 87.2%, groundedness 94.8%,
  expected-source recall 81.8%, and 22/24 scored at least 4. Both negative
  controls abstained correctly. The exact two-page report is
  `output/pdf/SG-IA_RAG_Two_Page_Executive_Summary.pdf`; detailed Italian and
  English-labelled audit records are stored in the run directory.
- RAG retrieval pipeline version 1.2 is the current post-baseline
  implementation. It uses a 100-token overlap for oversized split units,
  retrieves 24 semantic candidates, expands adjacent chunks only within the
  same document section, combines semantic rank with lexical question-facet
  coverage and diversity to rerank 8-10 final chunks, and performs at most one
  targeted retrieval retry when deterministic evidence-coverage checks find
  missing facets. The existing 24-document corpus was reindexed and verified
  as 496 Qdrant chunks with these chunking settings. Live regression checks
  answered the previous color, multi-year history, and activity-overlap failure
  patterns correctly.
- The full RAG v1.2 benchmark is
  `test_QA/RAG/results/20260813T115206Z-4a98ba`. All 25 chatbot and independent
  Claude Opus 5 judge calls completed. Results were 4.52/5 correctness, 90.4%
  required-point coverage, 96.2% groundedness, 89.1% expected-source recall,
  and 23/25 cases scoring at least 4. Both negative controls abstained correctly
  and no answerable case falsely abstained. The remaining incorrect case was a
  multi-source entity-list answer; one policy answer was partially correct.
  This is one stochastic comparison, not a statistical conclusion. The exact
  two-page report is
  `output/pdf/SG-IA_RAG_Two_Page_Executive_Summary_V1_2.pdf`; detailed Italian
  and English-labelled audit records are stored in the run directory.
- The complete RAG V2 benchmark is
  `test_QA/experiments/v2/RAG/results/20260826T071250Z-a66415`. It evaluated all
  100 V2 questions against the unchanged 24-document, 496-chunk Qdrant index,
  with no OCR, extraction, upload, re-ingestion, or ground-truth injection. The
  final recovery reused 99 hash-recorded chatbot responses from source run
  `20260826T065624Z-9bcf92`, retried only one transient RAG generation failure,
  and obtained 100/100 independent Claude Opus 5 judgments. Results are 4.06/5
  correctness, 76.5% required-point coverage, 94.3% groundedness, 87.8%
  expected-source recall, and 69/90 answerable cases scoring at least 4. All 10
  negative controls abstained correctly; nine answerable cases falsely
  abstained and 22/100 cases were flagged for unsupported claims. Detailed
  Italian and English READMEs beside the executive PDF preserve every question,
  RAG answer, ground truth, citation record, judge score, and explanation. The
  two-page report is
  `output/pdf/v2/RAG/1/SG-IA_RAG_V2_100Q_Executive_Summary.pdf`; its readable
  audits are stored beside it, with report navigation under
  `test_QA/experiments/v2/RAG/report/`. This is one stochastic run; the
  prior 25-question fixture is not directly comparable.
- `RAG/compose.yaml` independently starts Qdrant, FastAPI, and Streamlit with
  persistent named volumes. It supports either AWS environment credentials or
  a configurable local JSON credential file mounted read-only; no credentials
  enter the container image. The stack passed local container health checks on
  2026-08-13; backend unit/API tests use fakes and make no AWS calls.
- `WIKI/` already contains a FastAPI backend, Streamlit frontend, tests, and a
  persistent Markdown LLM Wiki proof of concept powered by AWS Bedrock.
- `WIKI/compose.yaml` independently runs that existing backend and frontend on
  host ports 8002 and 8503. It bind-mounts raw sources read-only and generated
  Wiki content read/write, excludes both from image builds, and can run beside
  the RAG Compose stack on 8001/8502.
- `WIKI/backend/AGENTS.md` is the operating contract specifically for wiki
  ingestion and Q&A. It remains authoritative inside `WIKI/backend/` and must
  be read before changing that component.
- The WIKI proof of concept was reset to a clean baseline on 2026-08-06: its
  previous fictitious test source and generated pages were removed. The current
  24-document operational subset is hash-matched to the shared `material/`
  corpus and declared by the private deployment manifest.
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
- WIKI Q&A uses `wiki-hybrid-section-search-v2`: deterministic page-level
  lexical search plus configurable Amazon Bedrock Titan Text Embeddings V2
  similarity over Markdown heading sections, fused by rank and aggregated into
  unique parent-page candidates. Section vectors are content-hash cached
  locally, refreshed after ingestion, and safely fall back to lexical search
  during an embedding outage. Raw-document chunks are never placed in this
  index and no vector database is required.
- Semantic sections are navigation hints, not answer evidence. The answer agent
  receives matching headings and excerpts, then reads the complete parent Wiki
  pages it chooses before answering or citing them. The Q&A path intentionally
  remains the direct Hybrid flow without the evidence-first ledger/verifier.
  API debug data exposes lexical/semantic ranks, selected parent pages, and
  matched sections for evaluation.
- WIKI read-only Q&A now permits four total complete answer-agent attempts for
  `BedrockError` and structured `AnswerSubmissionError` failures, with 0.25,
  0.5, and 1.0 second backoffs. Debug metadata separates transport and
  submission-failure counts. A fourth recoverable failure returns a sanitized
  503; unrelated validation/application failures and all write workflows remain
  non-retryable.
- WIKI chat responses include an optional 0-10 evidence-confidence score. A
  separate post-answer Bedrock pass verifies the answer against complete cited
  Wiki pages and exposes stable warning reasons. For the current POC this gate
  is advisory by default, so it does not replace the answer;
  `LLM_WIKI_ANSWER_GUARDRAIL_ENABLED=true` restores fail-closed enforcement.
  The UI displays the score while API debug metadata preserves verification and
  enforcement diagnostics.
- WIKI now routes trusted-manager chat actions into `fix_answer`,
  `update_knowledge`, or `add_knowledge`, always showing a preview and requiring
  explicit confirmation. New actions begin only with `/fix`, `/update`, or
  `/add`; all other messages remain normal Q&A, including later questions in
  the same session. The WIKI UI exposes dedicated action buttons and sends
  `/confirm` or `/cancel` for pending drafts. Additions create one stable
  subject source under `raw/manager-knowledge/`; its approved text is
  materialized verbatim into a deterministic source summary and optional
  canonical subject page, without generated embellishment.
  Updates atomically replace that same source, rewrite its existing source
  summary and every manifest-owned page, preserve stable subject/scope/period
  metadata, may delete a wholly obsolete source-exclusive page, and cannot
  increase the document or Wiki-page count; failed integration restores the
  prior source. Action history stays in
  `wiki/log.md`. Answer
  fixes create no raw knowledge: they must be verified from existing complete
  Wiki pages, then maintain one connected evidence page and create a non-indexed
  regression/audit
  record under `backend/feedback/answer-fixes/`. Drafts are in memory and all
  persistent paths remain manager-only POC behavior without authentication.
  An unsupported confirmed fix now writes nothing and becomes a visibly
  converted `update_knowledge` proposal requiring a second confirmation. The
  manager-action interpreter is instructed to preserve only correction facts
  explicitly supplied by the manager and not infer extra historical claims.
  The selected UI action is authoritative and cited pages provide update scope
  automatically. Explicit manager text is retained as audit input; an update
  previews and persists a complete merged current value rather than replacing
  the source with an incremental instruction. Recurring periods such as "every
  year" are confirmation-ready without an additional calendar year. A
  pre-commit fidelity check rejects derived rewrites that drop approved numeric
  or percentage details or confirmation conditions, or introduce numeric/date
  claims absent from all current page sources. Independent structured reviews
  reject unsupported inference both in the proposed merge and in staged Wiki
  prose. Explicit exact/verbatim requests must be preserved in a canonical
  page. A bounded repair turn runs before fail-closed rollback, and grounded
  answers retain material percentage and confirmation qualifiers, including
  stated timing and communication method. Invalid
  structured-answer attempts followed by free text can receive one fresh
  bounded submission reminder. Answer validation also rejects calculated or
  unsupported times, AM/PM, and timezone/local-time qualifiers and strips
  model-authored Sources sections. Bedrock transport is capped at a 60-second
  read timeout and one SDK attempt. Two temporary held-out manager knowledge
  lifecycles passed add/answer plus three update/answer rounds without page
  growth on 2026-08-17, after which all temporary artifacts were removed. The
  Sinergia mid-summer manager source and its
  two Wiki pages were repaired through the confirmed API flow on 2026-08-17 and
  now preserve the approved 100%-certainty statement and one-week email
  confirmation condition.
  Approved source knowledge changes must also enter RAG before comparison.
  Ambiguous or incomplete action details require clarification, and confirmation
  remains mandatory before any write. Insufficient-knowledge confidence is
  labeled as abstention confidence in the WIKI UI.
  While an action form is open, normal chat is disabled so a short human
  correction cannot accidentally enter Q&A. The form adds and deduplicates the
  action command automatically, preserves its text after an API failure, and
  unmarked `replace X with Y` prose receives safe UI guidance instead of a Q&A
  503. Orphan confirmation after a restart reports that no draft is pending.
- `test_QA/mateial/ground_truth_qa.json` is the initial shared evaluation fixture: 25
  Italian questions grounded in the WIKI raw corpus, with required answer
  points, source locators, evidence, and two insufficient-knowledge controls.
  `test_QA/mateial/README.md` documents the provisional comparison metrics.
- `test_QA/mateial/v2/ground_truth_qa_v2.json` is the separate shared V2
  fixture: 100 new Italian questions with `v2-qa-*` IDs, comprising 90
  answerable cases and 10 insufficient-knowledge controls. It preserves V1,
  has no exact V1 question reuse, covers all 24 hash-pinned comparison sources,
  and includes a validator for schema, source coverage, and corpus hashes.
- `test_QA/mateial/v2.1/ground_truth_qa_v2_1.json` is the focused WIKI
  paraphrase-generalization fixture: 56 answerable Italian questions derived
  exactly from the WIKI V2 baseline cases that originally scored below 4/5.
  Each case has a new `v2.1-qa-*` ID and a `parent_case_id`; only the question
  wording changes. Ground truths, required points, type, difficulty, sources,
  locators, and evidence are unchanged from the V2 parent. The set has zero
  exact V2 question overlap and no negative controls. It must be asked against
  the post-manager WIKI without further knowledge updates, and compared with
  the same 56 original baseline and exact-question post-update results rather
  than with the full 100-question aggregate.
- The completed WIKI V2 benchmark is
  `test_QA/experiments/v2/WIKI/results/20260825T102607Z-8c0929`. All 100 WIKI
  API responses and independent Claude Opus 5 judgments completed. Results are
  3.26/5 correctness, 53.1% required-point coverage, 80.0% groundedness, 84.4%
  expected-source recall, and 34/90 answerable cases scoring at least 4. All 10
  negative controls abstained correctly, while 11 answerable cases falsely
  abstained and 50/100 cases were flagged for unsupported claims. The run
  reused only hash-recorded accepted outputs while retrying transient WIKI and
  Bedrock failures; its current 65-page operational WIKI includes approved
  manager knowledge in addition to the 24-document benchmark corpus. The
  two-page report is
  `output/pdf/v2/WIKI/1/SG-IA_WIKI_V2_100Q_Executive_Summary.pdf`; detailed Italian and
  English audits are in the run directory. The V1 and V2 scores are not
  directly comparable because the fixtures differ.
- The first WIKI V2 knowledge-adaptation round completed on 2026-08-25 under
  `test_QA/experiments/v2/WIKI/results/knowledge-repair/20260825T112513Z-knowledge-repair/round1-no-ocr`.
  It sequentially asked all 56 answerable baseline cases scoring below 4/5,
  recorded their pre-update responses, atomically added each approved ground
  truth to its existing Wiki source-summary page, refreshed only the changed
  semantic section, and linted after every update. All 56 questions and updates
  completed with no API failures, no OCR or raw-document reads, 56 valid lint
  checks, and no Wiki-page growth (65 pages). This is an explicit
  knowledge-injection/adaptation run rather than an unbiased benchmark; the
  post-update evaluation is recorded separately.
- The complete post-manager WIKI V2 second round is
  `test_QA/experiments/v2/WIKI/results/post-manager-round2/20260825T163400Z-365426`.
  It preserves 100 fresh post-update chatbot answers and 99 immediately valid
  judgments from source run `20260825T154155Z-ceeef8`; one transient Claude
  Opus 5 service failure was recovered by retrying only that missing judgment.
  All 100 API responses and judgments completed. Compared with the original V2
  baseline, correctness moved from 3.26/5 to 4.61/5, answerable cases scoring
  at least 4 from 34/90 to 85/90, required-point coverage from 53.1% to 91.2%,
  groundedness from 80.0% to 94.9%, expected-source recall from 84.4% to 98.9%,
  and false abstentions from 11 to 0; all 10 negative controls still abstained
  correctly. On the exact 56 corrected cases, average correctness moved from
  2.18/5 to 4.66/5, 54 improved, one was unchanged, one declined, and 54/56
  now score at least 4. The two remaining below-4 cases are `v2-qa-034` and
  `v2-qa-089`. Executive PDFs are
  `output/pdf/v2/WIKI/2/SG-IA_WIKI_V2_Post_Manager_100Q_Executive_Summary.pdf`
  and
  `output/pdf/v2/WIKI/2/SG-IA_WIKI_V2_Manager_Corrections_56Q_Comparison.pdf`;
  detailed
  comparison artifacts are under
  `test_QA/experiments/v2/WIKI/report/post-manager-round2/`. This deliberately
  measures adaptation to disclosed ground truths, not unbiased generalization;
  held-out documents and paraphrased questions remain required.
- The completed WIKI V2.1 paraphrase-generalization run is
  `test_QA/experiments/v2.1/WIKI/results/20260826T085356Z-1f857b`. It asked all
  56 paraphrased questions against the unchanged post-manager Wiki, without OCR,
  re-ingestion, or further knowledge updates, and completed all 56 WIKI calls
  and independent Claude Opus 5 judgments without errors. Average correctness
  was 4.66/5; 55/56 scored at least 4; required-point coverage was 98.0%,
  groundedness 90.7%, expected-source recall 100%, and false abstentions 0.
  Compared with the same 56 exact-wording post-manager cases, the average stayed
  4.66/5, seven scores improved, 39 were unchanged, and ten declined; nine of
  the declines were 5-to-4 movements. The only below-4 case was `v2.1-qa-003`
  at 2/5. The judge flagged at least one unsupported claim in 23/56 answers, so
  strong correctness does not imply perfect evidence discipline. Complete
  Italian and English-labelled audits are in the run directory, comparison
  artifacts are under `test_QA/experiments/v2.1/WIKI/report/`, and the two-page
  report is
  `output/pdf/v2.1/WIKI/1/SG-IA_WIKI_V2_1_Paraphrase_56Q_Executive_Summary.pdf`.
  This evaluates generalization to new wording after ground-truth disclosure,
  not unseen knowledge or held-out documents.
- WIKI now supports unauthenticated POC user personalization and session-scoped
  conversation context. Private data lives only under ignored
  `WIKI/backend/user_data/<user_id>/`: durable `preferences.json`, readable
  `profile.md`, and exact per-session JSONL transcripts. The answer agent sees
  a bounded window of only the active session plus saved presentation
  preferences; neither is Wiki evidence or indexed knowledge. **New chat** and
  **Sign out** delete the active transcript while retaining the profile.
  Personalized API calls use `user_id` plus `session_id`; userless benchmark
  calls remain stateless and backward compatible. Both Compose entry points
  persist the private directory with a dedicated bind mount.
- Every normal message from an identified WIKI user now passes through a
  separate temperature-zero structured preference interpreter before grounded
  Q&A. It compares the message with all current preferences and returns
  `none`, `temporary`, `add`, `replace`, `remove`, or `clear`.
  Persistent changes require explicit intent plus confidence of at least 0.85;
  removals must name exact stored values and the backend performs the atomic
  mutation. Preference-only messages return without Wiki retrieval or
  citations; mixed messages update first and answer only the remaining factual
  question with the resolved preferences. Manager commands route first and
  userless benchmark calls avoid the detector and remain stateless. This linear
  routing does not currently require LangGraph.
- An initial isolated-user test on 2026-08-26 produced the intended replacement,
  but a later real UI run exposed a stochastic deletion-only classification for
  the same `never answer me in italian` command and left the profile empty.
  Preference decisions now include a separately validated intent kind:
  behavioral prohibitions require `persistent_behavior` plus `add` or
  `replace` and a retained actionable instruction; only explicit requests to
  delete remembered rules may use `memory_deletion` plus `remove` or
  `clear`. Inconsistent structured output is rejected and retried once.
  Post-fix live validation produced `persistent_behavior` plus `replace` for
  the conflict case, retained the prohibition, and a fresh grounded question
  consumed the saved preference and answered only in English. Temporary
  validation profiles and transcripts were removed.
- WIKI read-only Q&A treats both transient Bedrock transport failures and
  missing structured answer submissions as recoverable within the shared
  four-total-attempt ceiling. Unrelated validation errors and every
  manager/write workflow remain non-retryable.
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
- The `wiki-hybrid-section-search-v2` benchmark is
  `test_QA/WIKI/results/20260810-hybrid-section-v2-consolidated`. It contains
  one successful chatbot and Claude Opus 5 judgment for all 25 cases, using the
  first successful recovery result for two transient API failures. Standalone
  results are 3.88/5 correctness, 73.7% required-point coverage, 89.5%
  groundedness, 91.3% expected-source recall, 17/23 answerable cases scoring at
  least 4, and two false abstentions. All 40 recorded searches used
  `hybrid_section`. Detailed Italian/English records are in that run directory;
  the two-page report is
  `output/pdf/SG-IA_WIKI_Two_Page_Executive_Summary_Hybrid_Section_V2.pdf`.
- `comperision/` is an independently runnable Streamlit API client. It sends the
  same question and session ID concurrently to the WIKI and RAG `/chat`
  endpoints, presents WIKI on the left and RAG on the right, and exposes each
  system's status, answer, citations, timings, model metadata, and diagnostics.
  Partial failures are isolated so one unavailable backend does not hide the
  other's answer. Its Compose service publishes the UI on port 8504 and calls
  the two existing stacks through their host-published API ports. Comparison
  requests always use 10 final RAG evidence chunks with no UI control for this
  fixed evaluation setting.

## Decisions still required

Do not silently choose these during unrelated work. Resolve them when a task
depends on them, then update this file:

- LLM Wiki vs Google OKF vs evaluating both for the WIKI approach.
- Whether the current 24-document KnowledgeBase subset should become the
  formally frozen initial comparison corpus rather than the current operational
  subset.
- Model/provider and knowledge-generation pipeline for WIKI.
- Common API request/response schema.
- Final scoring weights.

## How to maintain this memory

Update this file whenever the user establishes or changes a durable goal,
constraint, architecture decision, major dependency, or milestone. Keep it
concise and factual. Move an item from "Decisions still required" into an
appropriate section once decided, and update the dated current-state section
after meaningful implementation changes.

Do not use this file as a task log or paste large code details into it. Detailed
component instructions belong in a nearer `AGENTS.md`; implementation history
belongs in version control.
