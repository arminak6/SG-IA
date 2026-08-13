"""Build detailed Markdown records and a two-page executive PDF for a RAG run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


NAVY = colors.HexColor("#193653")
BLUE = colors.HexColor("#2477B8")
GREEN = colors.HexColor("#2C946B")
AMBER = colors.HexColor("#E9AD32")
RED = colors.HexColor("#C83C3C")
INK = colors.HexColor("#16263A")
MUTED = colors.HexColor("#567095")
LINE = colors.HexColor("#CBD9E7")
PALE_BLUE = colors.HexColor("#E8F1FA")
PALE_GREEN = colors.HexColor("#E6F4ED")
PALE_AMBER = colors.HexColor("#FFF3D8")
PALE_RED = colors.HexColor("#FCE8E8")
LIGHT = colors.HexColor("#F5F8FB")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def blockquote(value: Any) -> str:
    text = str(value or "").strip() or "—"
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def outcome_label(value: str, *, italian: bool) -> str:
    labels = {
        "CORRECT": ("Corretta", "Correct"),
        "PARTIALLY_CORRECT": ("Parzialmente corretta", "Partially correct"),
        "INCORRECT": ("Errata", "Incorrect"),
        "FALSE_ABSTENTION": ("Astensione errata", "False abstention"),
        "EXPECTED_ABSTENTION": ("Astensione corretta", "Correct abstention"),
        "API_ERROR": ("Errore API", "API error"),
        "JUDGE_ERROR": ("Errore del giudice", "Judge error"),
    }
    pair = labels.get(value, (value, value))
    return pair[0] if italian else pair[1]


def detailed_markdown(
    records: Iterable[dict[str, Any]], *, run_id: str, italian: bool
) -> str:
    records = list(records)
    if italian:
        title = "# Valutazione del chatbot RAG: domande e risposte"
        intro = (
            f"Questo documento presenta i {len(records)} casi del benchmark, la risposta prodotta "
            f"dal chatbot RAG, la risposta ground truth e il giudizio di Claude Opus 5. "
            f"I dati provengono dal run `{run_id}`."
        )
        note = ""
    else:
        title = "# RAG chatbot evaluation: questions and answers"
        intro = (
            f"This document presents all {len(records)} benchmark cases, the RAG chatbot answer, "
            f"the ground-truth answer, and the Claude Opus 5 judgment from run `{run_id}`."
        )
        note = (
            "The benchmark questions and answers are preserved in their original Italian to avoid "
            "introducing translation changes into the auditable record.\n"
        )
    lines = [title, "", intro, "", note] if note else [title, "", intro]
    lines.extend(["", "## Navigazione rapida" if italian else "## Quick navigation", ""])
    for item in records:
        question = str(item.get("question", "")).replace("\n", " ")
        if len(question) > 92:
            question = question[:89] + "..."
        lines.append(f"- [{item['case_id']}: {question}](#{item['case_id']})")

    for item in records:
        chatbot = item.get("chatbot", {}) if isinstance(item.get("chatbot"), dict) else {}
        judgment = item.get("judgment", {}) if isinstance(item.get("judgment"), dict) else {}
        citations = chatbot.get("citations", []) if isinstance(chatbot.get("citations"), list) else []
        score = judgment.get("correctness_score", "—")
        outcome = outcome_label(str(item.get("primary_outcome", "")), italian=italian)
        headings = (
            ("Punteggio del giudice LLM", "Esito", "Domanda", "Risposta del chatbot RAG", "Risposta ground truth", "Fonti citate", "Diagnostica")
            if italian
            else ("LLM judge score", "Outcome", "Question (Italian)", "RAG chatbot answer (Italian)", "Ground-truth answer (Italian)", "Cited sources", "Diagnostics")
        )
        lines.extend(
            [
                "",
                "---",
                "",
                f"<a id=\"{item['case_id']}\"></a>",
                f"## {item['case_id']}",
                "",
                f"**{headings[0]}:** {score}/5  ",
                f"**{headings[1]}:** {outcome}",
                "",
                f"### {headings[2]}",
                "",
                blockquote(item.get("question")),
                "",
                f"### {headings[3]}",
                "",
                blockquote(chatbot.get("answer") or item.get("error")),
                "",
                f"### {headings[4]}",
                "",
                blockquote(item.get("ground_truth_answer")),
                "",
                f"### {headings[5]}",
                "",
            ]
        )
        if citations:
            for citation in citations:
                pages = ", ".join(str(page) for page in citation.get("page_numbers", [])) or "—"
                lines.append(
                    f"- `{citation.get('evidence_id', '—')}` — `{citation.get('source_path', '—')}`, "
                    f"page {pages}, score {citation.get('score', '—')}"
                )
        else:
            lines.append("- —")
        flags = ", ".join(item.get("diagnostic_flags", [])) or "—"
        explanation = judgment.get("correctness_explanation") or item.get("error") or "—"
        lines.extend(
            [
                "",
                f"### {headings[6]}",
                "",
                f"- Flags: `{flags}`",
                f"- Required-point coverage: `{item.get('required_point_coverage', '—')}`",
                f"- Groundedness: `{judgment.get('groundedness_score', '—')}`",
                f"- Expected-source recall: `{item.get('citation_metrics', {}).get('expected_source_recall', '—')}`",
                f"- Judge explanation: {explanation}",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


class Report:
    def __init__(self, target: Path) -> None:
        self.target = target
        self.width, self.height = A4
        self.c = canvas.Canvas(str(target), pagesize=A4)
        styles = getSampleStyleSheet()
        self.body = ParagraphStyle(
            "body", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.7,
            leading=9.6, textColor=INK, spaceAfter=0,
        )
        self.small = ParagraphStyle(
            "small", parent=self.body, fontSize=6.5, leading=7.6, textColor=MUTED,
        )
        self.cell = ParagraphStyle(
            "cell", parent=self.body, fontSize=6.6, leading=8.0,
        )
        self.center = ParagraphStyle(
            "center", parent=self.body, alignment=TA_CENTER, fontSize=7.0, leading=8.2,
        )
        self.left = 40
        self.right = self.width - 40

    def paragraph(self, text: str, x: float, y_top: float, width: float, height: float, style: ParagraphStyle | None = None) -> float:
        p = Paragraph(text, style or self.body)
        _, used = p.wrap(width, height)
        p.drawOn(self.c, x, y_top - used)
        return used

    def footer(self, page: int) -> None:
        self.c.setFont("Helvetica", 6.5)
        self.c.setFillColor(MUTED)
        self.c.drawString(self.left, 22, "SG-IA RAG benchmark | Management summary")
        self.c.drawRightString(self.right, 22, f"Page {page} of 2")

    def header(self, label: str) -> None:
        self.c.setFont("Helvetica-Bold", 7.5)
        self.c.setFillColor(MUTED)
        self.c.drawString(self.left, self.height - 26, label)
        self.c.setStrokeColor(LINE)
        self.c.line(self.left, self.height - 35, self.right, self.height - 35)

    def title(self, text: str, y: float, size: float = 24) -> None:
        self.c.setFont("Helvetica-Bold", size)
        self.c.setFillColor(NAVY)
        self.c.drawString(self.left, y, text)

    def section_title(self, text: str, y: float, size: float = 15) -> None:
        self.c.setFont("Helvetica-Bold", size)
        self.c.setFillColor(BLUE)
        self.c.drawString(self.left, y, text)

    def band(self, y_top: float, height: float, fill: colors.Color, stroke: colors.Color, label: str, text: str) -> None:
        self.c.setFillColor(fill)
        self.c.setStrokeColor(stroke)
        self.c.rect(self.left, y_top - height, self.right - self.left, height, fill=1, stroke=1)
        self.c.setFont("Helvetica", 7.2)
        self.c.setFillColor(stroke)
        self.c.drawString(self.left + 10, y_top - 15, label)
        self.paragraph(text, self.left + 78, y_top - 7, self.right - self.left - 90, height - 10)

    def kpis(self, y_top: float, items: list[tuple[str, str, colors.Color]]) -> None:
        width = (self.right - self.left) / len(items)
        for index, (value, label, tint) in enumerate(items):
            x = self.left + index * width
            self.c.setFillColor(tint)
            self.c.setStrokeColor(LINE)
            self.c.rect(x, y_top - 54, width, 54, fill=1, stroke=1)
            self.c.setFillColor(GREEN if index in {0, 1, 3} else BLUE)
            self.c.setFont("Helvetica-Bold", 17)
            self.c.drawCentredString(x + width / 2, y_top - 23, value)
            self.c.setFillColor(MUTED)
            self.c.setFont("Helvetica-Bold", 6.4)
            self.c.drawCentredString(x + width / 2, y_top - 41, label)

    def table(self, x: float, y_top: float, widths: list[float], rows: list[list[str]], row_heights: list[float]) -> None:
        y = y_top
        for row_index, (row, height) in enumerate(zip(rows, row_heights)):
            fill = LIGHT if row_index == 0 else colors.white
            self.c.setFillColor(fill)
            self.c.setStrokeColor(LINE)
            cursor = x
            for value, width in zip(row, widths):
                self.c.rect(cursor, y - height, width, height, fill=1, stroke=1)
                style = ParagraphStyle(
                    f"cell-{row_index}-{cursor}", parent=self.cell,
                    fontName="Helvetica-Bold" if row_index == 0 else "Helvetica",
                )
                self.paragraph(value, cursor + 5, y - 4, width - 10, height - 7, style)
                cursor += width
            y -= height

    def finish_page(self, page: int) -> None:
        self.footer(page)
        self.c.showPage()

    def save(self) -> None:
        self.c.save()


def draw_report(run_dir: Path, output_pdf: Path, records: list[dict[str, Any]], summary: dict[str, Any], manifest: dict[str, Any]) -> None:
    judged = [item for item in records if isinstance(item.get("judgment"), dict)]
    scores = Counter(int(item["judgment"]["correctness_score"]) for item in judged)
    wiki_summary_path = Path(__file__).resolve().parents[2] / "test_QA" / "WIKI" / "results" / "20260810-hybrid-section-v2-consolidated" / "summary.json"
    wiki_summary = load_json(wiki_summary_path) if wiki_summary_path.is_file() else None
    report = Report(output_pdf)
    c = report.c

    report.header("SG-IA  |  RAG EXECUTIVE EVALUATION")
    report.title("SG-IA RAG Chatbot", report.height - 66)
    c.setFont("Helvetica", 10)
    c.setFillColor(BLUE)
    c.drawString(report.left, report.height - 87, "Two-page executive summary | 25 Italian benchmark questions | 13 August 2026")
    report.band(
        report.height - 100, 30, PALE_AMBER, colors.HexColor("#E49B12"), "DECISION",
        "Promising for a controlled internal pilot with human review; one structured-answer failure remains before unattended use.",
    )
    correctness = summary["correctness"]
    grounding = summary["grounding_and_sources"]
    report.kpis(
        report.height - 138,
        [
            (f"{correctness['average_score_1_to_5']:.2f}/5", "AVERAGE CORRECTNESS · 24 JUDGED", PALE_BLUE),
            ("22/24", "JUDGED CASES SCORING 4–5", PALE_GREEN),
            (f"{100 * correctness['average_required_point_coverage']:.1f}%", "REQUIRED-POINT COVERAGE", PALE_BLUE),
            (f"{100 * grounding['average_groundedness']:.1f}%", "GROUNDEDNESS", PALE_GREEN),
        ],
    )

    y = report.height - 214
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(report.width / 2, y, "Correctness score distribution")
    chart_left, chart_bottom, chart_width, chart_height = 78, y - 132, report.width - 118, 94
    max_count = max(max(scores.values(), default=1), 1)
    for tick in range(0, max_count + 1, 2):
        yy = chart_bottom + chart_height * tick / max_count
        c.setStrokeColor(LINE)
        c.line(chart_left, yy, chart_left + chart_width, yy)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 6)
        c.drawRightString(chart_left - 8, yy - 2, str(tick))
    palette = {1: RED, 2: colors.HexColor("#D75B3B"), 3: AMBER, 4: colors.HexColor("#5A9AC7"), 5: GREEN}
    slot = chart_width / 5
    for score in range(1, 6):
        count = scores.get(score, 0)
        height = chart_height * count / max_count
        bar_width = slot * 0.6
        x = chart_left + (score - 1) * slot + slot * 0.2
        c.setFillColor(palette[score])
        c.rect(x, chart_bottom, bar_width, height, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 7)
        c.drawCentredString(x + bar_width / 2, chart_bottom + height + 5, str(count))
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 7)
        c.drawCentredString(x + bar_width / 2, chart_bottom - 13, str(score))
    c.setFont("Helvetica-Oblique", 6.8)
    c.setFillColor(MUTED)
    c.drawCentredString(report.width / 2, chart_bottom - 29, "Claude Opus 5 judge score (1–5); one API failure is excluded from this chart.")

    report.section_title("What the evaluation showed", chart_bottom - 55)
    table_y = chart_bottom - 65
    report.table(
        report.left,
        table_y,
        [(report.right - report.left) / 2] * 2,
        [
            ["Strong experience", "Observed limitation"],
            [
                "Direct facts: 4.44/5<br/>Procedures: 4.17/5<br/>Unknown-information controls: 2/2 correct",
                "Multi-source expected-source recall: 33.3%<br/>Unsupported-claim flags: 5/24<br/>Structured-answer API failures: 1/25",
            ],
        ],
        [20, 43],
    )
    report.band(
        table_y - 72, 38, PALE_RED, RED, "PRIMARY RISK",
        "Retrieval and generation are strong overall, but multi-source coverage and structured-output reliability can still produce incomplete or unavailable answers.",
    )
    report.finish_page(1)

    report.section_title("Performance profile and next decision", report.height - 42, 16)
    y_top = report.height - 75
    c.setFont("Helvetica-Bold", 10.5)
    c.setFillColor(NAVY)
    c.drawString(90, y_top, "Quality by question type")
    c.drawString(344, y_top, "RAG vs WIKI (directional)")
    groups = summary["by_question_type"]
    labels = [
        ("Unknown controls", groups.get("unanswerable", {}).get("average_correctness_score")),
        ("Policy application", groups.get("policy_application", {}).get("average_correctness_score")),
        ("Direct facts", groups.get("single_source_fact", {}).get("average_correctness_score")),
        ("Comparisons", groups.get("single_source_comparison", {}).get("average_correctness_score")),
        ("Procedures", groups.get("procedure", {}).get("average_correctness_score")),
        ("Multi-source", groups.get("multi_source_synthesis", {}).get("average_correctness_score")),
    ]
    bar_x, label_x, value_x = 122, 45, 290
    for index, (label, value) in enumerate(labels):
        value = float(value or 0)
        yy = y_top - 28 - index * 28
        c.setFont("Helvetica", 6.8)
        c.setFillColor(MUTED)
        c.drawRightString(label_x + 70, yy + 3, label)
        c.setFillColor(GREEN if value >= 4 else AMBER if value >= 3 else RED)
        c.rect(bar_x, yy - 3, 130 * value / 5, 12, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 6.5)
        c.drawString(bar_x + 130 * value / 5 + 5, yy, f"{value:.2f}")

    compare_metrics = [
        ("Correctness /5", correctness["average_score_1_to_5"], wiki_summary["correctness"]["average_score_1_to_5"] if wiki_summary else None),
        ("Point coverage", 100 * correctness["average_required_point_coverage"], 100 * wiki_summary["correctness"]["average_required_point_coverage"] if wiki_summary else None),
        ("Groundedness", 100 * grounding["average_groundedness"], 100 * wiki_summary["grounding_and_sources"]["average_groundedness"] if wiki_summary else None),
        ("Source recall", 100 * grounding["average_expected_source_recall"], 100 * wiki_summary["grounding_and_sources"]["average_expected_source_recall"] if wiki_summary else None),
    ]
    cx, cw = 342, 190
    for index, (label, rag_value, wiki_value) in enumerate(compare_metrics):
        yy = y_top - 25 - index * 38
        c.setFont("Helvetica", 6.7)
        c.setFillColor(MUTED)
        c.drawString(cx, yy + 13, label)
        max_value = 5 if label.endswith("/5") else 100
        c.setFillColor(colors.HexColor("#D7E3EE"))
        c.rect(cx, yy, cw * float(wiki_value or 0) / max_value, 7, fill=1, stroke=0)
        c.setFillColor(BLUE)
        c.rect(cx, yy - 9, cw * float(rag_value) / max_value, 7, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 6.2)
        c.setFillColor(INK)
        suffix = "" if label.endswith("/5") else "%"
        c.drawRightString(cx + cw, yy + 1, f"W {float(wiki_value or 0):.1f}{suffix}")
        c.drawRightString(cx + cw, yy - 8, f"R {float(rag_value):.1f}{suffix}")
    c.setFont("Helvetica-Oblique", 6.5)
    c.setFillColor(MUTED)
    c.drawCentredString(report.width / 2, y_top - 188, "Directional only: single stochastic runs; RAG quality metrics exclude one API failure.")

    report.section_title("Operational experience", y_top - 218, 12)
    latency = summary["latency_ms"]
    report.kpis(
        y_top - 228,
        [
            ("24/25", "COMPLETED AND JUDGED", PALE_AMBER),
            ("2/2", "CORRECT NEGATIVE CONTROLS", PALE_GREEN),
            (f"{latency['server_median'] / 1000:.2f} s", "MEDIAN SERVER LATENCY", PALE_BLUE),
            (f"{latency['server_p95'] / 1000:.2f} s", "P95 SERVER LATENCY", PALE_BLUE),
        ],
    )

    report.section_title("Recommended action plan", y_top - 302, 12)
    report.table(
        report.left,
        y_top - 312,
        [40, 272, report.right - report.left - 312],
        [
            ["Priority", "Action", "Success measure"],
            ["P0", "Add a provider-compatible structured-output fallback and retain bounded retries.", "100% completion across three repeated 25-case runs"],
            ["P1", "Improve multi-source retrieval with query expansion or diversified parent-document selection.", "At least 95% expected-source recall"],
            ["P1", "Tighten claim-to-citation validation before returning the final answer.", "Unsupported-claim flag rate at or below 5%"],
            ["P2", "Validate on paraphrased questions and held-out documents with human review.", "Stable quality without benchmark-specific tuning"],
        ],
        [18, 31, 31, 31, 31],
    )
    report.band(
        y_top - 468, 42, PALE_GREEN, GREEN, "PILOT BOUNDARY",
        "Allow low-risk internal factual and procedural queries with visible citations. Require human verification for policy, safety, financial, HR, and multi-source decisions.",
    )
    method = (
        f"Method: one response per question from RAG / GPT-OSS 20B; Titan V2 512-dimensional embeddings; "
        f"top-k 8 over {manifest['corpus_manifest']['document_count']} documents and "
        f"{manifest['corpus_manifest']['chunk_count']} chunks. Independently judged by Claude Opus 5 on Amazon Bedrock. "
        f"Run {run_dir.name}; 24/25 answers completed after bounded retries. Chatbot usage: "
        f"{summary['usage']['chatbot'].get('totalTokens', 0) / 1000:.0f}K tokens; judge usage: "
        f"{summary['usage']['judge'].get('totalTokens', 0) / 1000:.0f}K tokens. Costs not inferred without pinned pricing."
    )
    report.paragraph(method, report.left, 71, report.right - report.left, 40, report.small)
    report.finish_page(2)
    report.save()
    pages = len(PdfReader(str(output_pdf)).pages)
    if pages != 2:
        raise RuntimeError(f"Expected exactly two pages, generated {pages}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-pdf", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    records = load_jsonl(run_dir / "results.jsonl")
    summary = load_json(run_dir / "summary.json")
    manifest = load_json(run_dir / "run_manifest.json")
    (run_dir / "README.md").write_text(
        detailed_markdown(records, run_id=run_dir.name, italian=True), encoding="utf-8", newline="\n"
    )
    (run_dir / "README_EN.md").write_text(
        detailed_markdown(records, run_id=run_dir.name, italian=False), encoding="utf-8", newline="\n"
    )
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    draw_report(run_dir, args.output_pdf.resolve(), records, summary, manifest)
    print(args.output_pdf.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
