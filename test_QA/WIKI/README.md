# WIKI chatbot benchmark

This program sends the 25 Italian questions in
`../mateial/ground_truth_qa.json` to the running WIKI API and evaluates every
answer with a separately configured, high-capability Amazon Bedrock model.

It measures point-level semantic correctness, 1–5 correctness, groundedness
against the Wiki pages actually read, expected-source citation recall,
abstention, latency, token usage, optional estimated cost, and diagnostic
failure categories. It does not use RAG-only ranked-retrieval metrics.

## 1. Start the WIKI backend

From `WIKI/`:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The evaluator checks `/health` and `/wiki/lint` before making paid calls. All
24 source documents should already be ingested and lint must be valid.

## 2. Install and configure the evaluator

From `test_QA/WIKI/`:

```powershell
python -m pip install -r requirements.txt
Copy-Item config.example.json config.json
```

Edit `config.json` and replace `judge.model_id` with the Bedrock model or
cross-region inference-profile ID chosen for judging. Prefer a strong model
different from the chatbot model. The selected model should support Bedrock
Converse and preferably tool use.

`judge.temperature` is `null` by default because some frontier Bedrock models,
including Claude Opus 5, reject that deprecated parameter. For a model that
supports it, set it to `0.0` for deterministic judging.

The example reuses `../../WIKI/aws_credentials.json`. Explicit keys are
optional; when absent, Boto3 uses its normal AWS credential chain. Never add
keys to `config.json`, which is ignored by Git.

The judge model can also be supplied without editing the file:

```powershell
$env:BEDROCK_JUDGE_MODEL_ID = "your-model-or-inference-profile-id"
$env:AWS_REGION = "us-east-1"
```

Pricing is intentionally not hardcoded because it depends on the selected
model, region, and inference mode. Fill in the four per-million-token fields if
cost estimates are wanted; otherwise cost fields are reported as `null`.

## 3. Run

Start with one answerable and one unanswerable case:

```powershell
python evaluate_wiki.py --case-id qa-001 --case-id qa-024
```

Run all 25 questions once:

```powershell
python evaluate_wiki.py
```

Run each question three times for stability measurement:

```powershell
python evaluate_wiki.py --repetitions 3
```

Useful overrides include `--api-url`, `--dataset`, `--wiki-root`,
`--judge-model-id`, `--aws-region`, `--limit`, and `--output-dir`. Run
`python evaluate_wiki.py --help` for the complete list.

Questions run sequentially so client and server latency are comparable. A
non-zero exit code means at least one API or judge call failed, but completed
records are still preserved. Transient `429`/`5xx` chatbot failures are retried
up to three times with backoff, and a malformed judge result is retried once;
both attempt counts are saved in the detailed results.

## Results

Every run creates `results/<UTC timestamp>-<suffix>/` containing:

- `results.jsonl`: complete response, judgment, evidence metadata, and failure
  diagnostics for every question/repetition.
- `summary.csv`: compact per-result table for spreadsheet analysis.
- `summary.json`: global correctness, grounding, abstention, latency, usage,
  and cost metrics.
- `run_manifest.json`: dataset/corpus/prompt hashes, model IDs, configuration,
  and preflight state needed for reproducibility. Credentials are never stored.

Wiki page text is sent to the judge for groundedness, but the output stores
only page paths, hashes, character counts, and truncation flags rather than
duplicating the corpus.

## Offline tests

These tests do not call the WIKI API or AWS:

```powershell
python -m unittest discover -s tests -v
```
