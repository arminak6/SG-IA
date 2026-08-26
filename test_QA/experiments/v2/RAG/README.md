# RAG V2 experiment

The completed 100-question RAG evaluation is stored under
`results/20260826T071250Z-a66415/`. Its `results.jsonl`, `summary.json`,
`summary.csv`, and `run_manifest.json` are the authoritative machine-readable
artifacts.

The readable Italian and English audits and presentation-ready PDF are stored
under `output/pdf/v2/RAG/1/`. They contain every question, RAG answer,
ground-truth answer, returned citations, judge score, outcome, and judge
explanation.

This run used only the existing 24-document, 496-chunk Qdrant index. No OCR,
extraction, upload, re-ingestion, or ground-truth injection was performed.
