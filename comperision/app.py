"""Side-by-side Streamlit chatbot for the RAG and LLM Wiki APIs."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

import streamlit as st

from api_client import (
    DEFAULT_RAG_API_URL,
    DEFAULT_WIKI_API_URL,
    ask_both,
    check_both,
)


st.set_page_config(
    page_title="SG-IA · RAG vs LLM Wiki",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1500px; padding-top: 2rem;}
      [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, .22);
        border-radius: .65rem;
        padding: .65rem .8rem;
      }
      .comparison-subtitle {color: #6b7280; margin-top: -.8rem;}
      .answer-label {
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: #6b7280;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _new_session_id() -> str:
    return f"comparison-{uuid4().hex}"


if "comparison_turns" not in st.session_state:
    st.session_state.comparison_turns = []
if "comparison_session_id" not in st.session_state:
    st.session_state.comparison_session_id = _new_session_id()


def _format_ms(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    if value >= 1000:
        return f"{value / 1000:.2f} s"
    return f"{value:.0f} ms"


def _status_label(value: Any) -> str:
    return str(value or "unknown").replace("_", " ").title()


def _render_health(label: str, payload: Mapping[str, Any] | None) -> None:
    if not payload:
        st.caption(f"⚪ {label}: not checked")
        return
    if payload.get("healthy"):
        st.caption(f"🟢 {label}: {_status_label(payload.get('status'))}")
    else:
        st.caption(f"🔴 {label}: {payload.get('error') or payload.get('status')}")


def _render_wiki_citations(citations: list[Mapping[str, Any]]) -> None:
    for index, citation in enumerate(citations, start=1):
        st.markdown(f"**{index}. Wiki page**")
        st.text(str(citation.get("wiki_path", "Unknown Wiki page")))
        sources = citation.get("source_paths", [])
        if isinstance(sources, list) and sources:
            st.caption("Source documents")
            for source in sources:
                st.text(f"• {source}")
        if index != len(citations):
            st.divider()


def _render_rag_citations(citations: list[Mapping[str, Any]]) -> None:
    for index, citation in enumerate(citations, start=1):
        evidence_id = citation.get("evidence_id") or f"Evidence {index}"
        title = citation.get("title") or citation.get("source_path") or "Source"
        st.markdown(f"**{evidence_id} · {title}**")
        st.text(str(citation.get("source_path", "Unknown source")))

        details: list[str] = []
        pages = citation.get("page_numbers", [])
        if isinstance(pages, list) and pages:
            details.append("pages " + ", ".join(str(page) for page in pages))
        score = citation.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            details.append(f"score {score:.3f}")
        if details:
            st.caption(" · ".join(details))

        excerpt = citation.get("excerpt")
        if excerpt:
            st.write(str(excerpt))
        if index != len(citations):
            st.divider()


def _render_answer(label: str, result: Mapping[str, Any], approach: str) -> None:
    st.markdown(f'<div class="answer-label">{label}</div>', unsafe_allow_html=True)
    st.subheader("LLM Wiki" if approach == "wiki" else "RAG")

    error = result.get("error")
    if error:
        st.error(str(error))
        st.caption(f"Client wait: {_format_ms(result.get('client_elapsed_ms'))}")
        return

    status = str(result.get("status", "unknown"))
    if status.casefold() in {"answered", "ok"}:
        st.success(_status_label(status), icon="✅")
    else:
        st.warning(_status_label(status), icon="⚠️")

    answer = str(result.get("answer", "")).strip()
    if answer:
        st.markdown(answer)
    else:
        st.warning("The backend returned an empty answer.")

    metric_columns = st.columns(3)
    metric_columns[0].metric("Server", _format_ms(result.get("server_latency_ms")))
    metric_columns[1].metric("End to end", _format_ms(result.get("client_elapsed_ms")))
    confidence = result.get("confidence_score")
    confidence_label = (
        f"{confidence:.1f}/10" if isinstance(confidence, (int, float)) else "—"
    )
    metric_columns[2].metric("Confidence", confidence_label)

    model_id = result.get("model_id")
    if model_id:
        st.caption(f"Model: {model_id}")

    citations = result.get("citations", [])
    if isinstance(citations, list):
        with st.expander(f"Evidence and citations ({len(citations)})"):
            if not citations:
                st.caption("No citations were returned.")
            elif approach == "wiki":
                _render_wiki_citations(citations)
            else:
                _render_rag_citations(citations)

    manager_action = result.get("manager_action") or result.get("correction")
    if manager_action:
        st.warning(
            "The Wiki proposed a manager action. This comparison client does not "
            "confirm or persist knowledge changes."
        )

    diagnostics = {
        "usage": result.get("usage", {}),
        "timings": result.get("timings", {}),
        "debug": result.get("debug", {}),
    }
    if result.get("embedding_model_id"):
        diagnostics["embedding_model_id"] = result["embedding_model_id"]
    if manager_action:
        diagnostics["manager_action"] = manager_action
    with st.expander("Diagnostics"):
        st.json(diagnostics)


def _render_turn(turn: Mapping[str, Any]) -> None:
    with st.chat_message("user"):
        st.markdown(str(turn["question"]))
        asked_at = turn.get("asked_at")
        if asked_at:
            st.caption(str(asked_at))

    wiki_column, rag_column = st.columns(2, gap="large", border=True)
    with wiki_column:
        _render_answer("Left · Knowledge Wiki", turn["wiki"], "wiki")
    with rag_column:
        _render_answer("Right · Vector Retrieval", turn["rag"], "rag")


st.title("RAG vs LLM Wiki")
st.markdown(
    '<p class="comparison-subtitle">Ask once. The same question is sent to both '
    "backends concurrently and the grounded answers are shown side by side.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Connections")
    rag_api_url = st.text_input(
        "RAG API",
        value=os.getenv("RAG_API_URL", DEFAULT_RAG_API_URL),
        help="FastAPI base URL; /chat and /health are called.",
    )
    wiki_api_url = st.text_input(
        "LLM Wiki API",
        value=os.getenv("WIKI_API_URL", DEFAULT_WIKI_API_URL),
        help="FastAPI base URL; /chat and /health are called.",
    )

    if st.button("Check both APIs", use_container_width=True):
        with st.spinner("Checking both backends…"):
            health = check_both(
                rag_api_url=rag_api_url,
                wiki_api_url=wiki_api_url,
            )
        st.session_state.comparison_health = {
            name: result.to_dict() for name, result in health.items()
        }

    saved_health = st.session_state.get("comparison_health", {})
    _render_health("LLM Wiki", saved_health.get("wiki"))
    _render_health("RAG", saved_health.get("rag"))

    st.divider()
    st.caption("RAG final evidence chunks")
    rag_top_k = st.slider(
        "Top K",
        min_value=8,
        max_value=10,
        value=10,
        label_visibility="collapsed",
        help="The RAG v1.2 API accepts 8–10 final reranked chunks.",
    )

    if st.button("New conversation", use_container_width=True):
        st.session_state.comparison_turns = []
        st.session_state.comparison_session_id = _new_session_id()
        st.rerun()

    st.caption(
        "This UI performs no ingestion and reads no knowledge store directly. "
        "It only calls the two backend APIs."
    )

for saved_turn in st.session_state.comparison_turns:
    _render_turn(saved_turn)

question = st.chat_input("Ask the RAG and LLM Wiki the same question…", max_chars=10_000)
if question:
    with st.spinner("RAG and LLM Wiki are answering concurrently…"):
        results = ask_both(
            question,
            session_id=st.session_state.comparison_session_id,
            rag_api_url=rag_api_url,
            wiki_api_url=wiki_api_url,
            rag_top_k=rag_top_k,
        )

    turn = {
        "question": question.strip(),
        "asked_at": datetime.now(timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%d %H:%M:%S %Z"),
        "wiki": results["wiki"].to_dict(),
        "rag": results["rag"].to_dict(),
    }
    st.session_state.comparison_turns.append(turn)
    _render_turn(turn)
