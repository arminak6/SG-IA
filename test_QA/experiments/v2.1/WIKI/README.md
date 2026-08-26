# WIKI V2.1 experiment

V2.1 contains 56 natural Italian paraphrases of the answerable V2 baseline cases that originally scored below 4/5. Ground truths and required answer points are inherited from the parent V2 cases.

- Completed run: `results/20260826T085356Z-1f857b`
- Complete Italian audit: `results/20260826T085356Z-1f857b/README.md`
- English-labelled audit: `results/20260826T085356Z-1f857b/README_EN.md`
- Comparison artifacts: `report/`
- Executive PDF: `output/pdf/v2.1/WIKI/1/SG-IA_WIKI_V2_1_Paraphrase_56Q_Executive_Summary.pdf`

Headline result: 4.66/5 average correctness; 55/56 scored at least 4; required-point coverage 98.0%; groundedness 90.7%; expected-source recall 100.0%. All 56 API calls and judgments completed.

Interpretation: aggregate performance was retained under paraphrasing after the manager knowledge adaptation, but this is not an unseen-knowledge benchmark because all 56 ground truths had previously been disclosed to the Wiki.
