# SG-IA benchmark experiment V2

This directory keeps the 100-question V2 comparison experiment isolated from
the original 25-question benchmark.

## Shared input

- Dataset: `test_QA/mateial/v2/ground_truth_qa_v2.json`
- Corpus manifest: `deployment/corpus-manifest.json`
- Repetitions: one answer per case for the initial comparison run
- Judge: an independently configured Amazon Bedrock model

## Layout

- `WIKI/results/`: timestamped raw WIKI API responses, judge records, metrics,
  and reproducibility manifests.
- `WIKI/report/`: WIKI report metadata and supporting summaries.
- `RAG/results/`: timestamped raw RAG API responses, judge records, metrics,
  and reproducibility manifests.
- `RAG/report/`: RAG report metadata.

The presentation-ready PDF is published under `output/pdf/` so it remains next
to the earlier executive evaluation reports. Generated results and reports are
private evaluation artifacts and are excluded by the repository `.gitignore`.

## Completed WIKI run

- Final run: `WIKI/results/20260825T102607Z-8c0929`
- API and judge completion: 100/100
- Average correctness: 3.26/5
- Answerable cases scoring 4-5: 34/90
- Required-point coverage: 53.1%
- Groundedness: 80.0%
- Expected-source recall: 84.4%
- Correct insufficient-knowledge controls: 10/10
- Executive PDF: `output/pdf/v2/WIKI/1/SG-IA_WIKI_V2_100Q_Executive_Summary.pdf`

The run was recovered without changing accepted outputs: successful chatbot
responses and judgments were reused with source-run hashes, while only failed
API calls and transient judge service errors were retried. Detailed English and
Italian audits are stored inside the final timestamped run directory. Both
README editions contain all 100 questions, LLM answers, ground-truth answers,
judge scores, point-level verdicts, citations, and judge explanations.

## Completed RAG run

- Final run: `RAG/results/20260826T071250Z-a66415`
- API and judge completion: 100/100
- Average correctness: 4.06/5
- Answerable cases scoring 4-5: 69/90
- Required-point coverage: 76.5%
- Groundedness: 94.3%
- Expected-source recall: 87.8%
- Correct insufficient-knowledge controls: 10/10
- Executive PDF and readable audits: `output/pdf/v2/RAG/1/`

The RAG run used only the existing 24-document, 496-chunk Qdrant index. No OCR,
extraction, upload, re-ingestion, or ground-truth injection was performed.
