# RAG benchmark

This evaluator runs the same 25-case Italian fixture used by the WIKI benchmark
against the RAG `/chat` API. It stores the complete chatbot response, evaluates
required-point correctness and groundedness with an independently configured
Amazon Bedrock judge, measures expected-source recall, and records latency and
reproducibility metadata.

The RAG API uses `insufficient_evidence`; the shared fixture calls the equivalent
state `insufficient_knowledge`. The evaluator normalizes only that status label.
For source recall it compares normalized source basenames because the RAG API
does not disclose corpus directory paths. The judge receives only cited chunks
from `debug.retrieved_chunks`, not uncited retrieval candidates.

Copy `config.example.json` to the ignored `config.json`, set a judge model and
credentials, start the RAG Compose stack, then run:

```powershell
python test_QA/RAG/evaluate_rag.py --config test_QA/RAG/config.json
```

Each run creates a timestamped directory under `results/` containing
`results.jsonl`, `summary.json`, `summary.csv`, and `run_manifest.json`.
Credentials, private fixtures, and generated benchmark outputs remain ignored.
