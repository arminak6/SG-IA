"""Focused Streamlit client for RAG document upload and grounded chat."""

from __future__ import annotations

import os
from typing import Any, Mapping
from uuid import uuid4

import streamlit as st

from api_client import RagApiClient, RagApiError


SUPPORTED_EXTENSIONS = ["pdf", "docx", "pptx", "md", "txt", "csv", "json"]
TERMINAL_JOB_STATUSES = {"completed", "failed"}


st.set_page_config(
    page_title="SG-IA RAG",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 980px;
            padding-top: 2.6rem;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.6rem;
        }
        .app-kicker {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
        }
        .upload-note {
            color: #64748b;
            font-size: 0.82rem;
            margin-top: -0.4rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _format_ms(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    if value >= 1000:
        return f"{value / 1000:.1f} s"
    return f"{value:.0f} ms"


def _render_ingestion(job: Mapping[str, Any]) -> None:
    status = str(job.get("status", "unknown")).casefold()
    filename = str(job.get("filename") or "Document")

    if status == "completed":
        chunks = int(job.get("chunk_count") or 0)
        st.success(f"{filename} is ready for chat · {chunks} indexed chunks")
    elif status == "failed":
        st.error(str(job.get("error") or f"Indexing failed for {filename}."))
    else:
        stage = str(job.get("stage") or status).replace("_", " ").title()
        st.info(f"Indexing {filename} · {stage}")

    with st.expander("Ingestion details"):
        st.json(dict(job))

    if status not in TERMINAL_JOB_STATUSES:
        st.button("Refresh indexing status", use_container_width=True)


def _render_citations(citations: list[Mapping[str, Any]]) -> None:
    if not citations:
        st.caption("No source citations were returned.")
        return

    for index, citation in enumerate(citations, start=1):
        evidence_id = str(citation.get("evidence_id") or f"Evidence {index}")
        source = str(citation.get("source_path") or "Unknown source")
        pages = citation.get("page_numbers", [])
        page_label = ""
        if isinstance(pages, list) and pages:
            page_label = " · pages " + ", ".join(str(page) for page in pages)
        score = citation.get("score")
        score_label = ""
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            score_label = f" · score {score:.3f}"

        st.markdown(f"**{evidence_id} · {source}{page_label}{score_label}**")
        headings = citation.get("heading_path", [])
        if isinstance(headings, list) and headings:
            st.caption(" > ".join(str(heading) for heading in headings))
        excerpt = str(citation.get("excerpt") or "").strip()
        if excerpt:
            st.write(excerpt)
        if index != len(citations):
            st.divider()


def _render_chat_message(message: Mapping[str, Any]) -> None:
    with st.chat_message(str(message["role"])):
        status = str(message.get("status") or "")
        if status and status != "answered":
            st.warning(status.replace("_", " ").title())
        st.markdown(str(message["content"]))

        confidence_score = message.get("confidence_score")
        if isinstance(confidence_score, (int, float)) and not isinstance(
            confidence_score, bool
        ):
            st.markdown(f"**Confidence score: {float(confidence_score):.1f}/10**")

        if message.get("role") != "assistant" or message.get("is_greeting"):
            return

        citations = message.get("citations", [])
        citation_count = len(citations) if isinstance(citations, list) else 0
        timings = message.get("timings", {})
        elapsed = timings.get("total_ms") if isinstance(timings, Mapping) else None
        summary = [part for part in (_format_ms(elapsed), f"{citation_count} source(s)") if part]
        if summary:
            st.caption(" · ".join(summary))

        with st.expander(f"Sources ({citation_count})"):
            _render_citations(citations if isinstance(citations, list) else [])

        diagnostics = message.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            with st.expander("Technical details"):
                st.json(dict(diagnostics))


default_api_url = os.getenv("RAG_API_URL", "http://127.0.0.1:8001")

if "rag_api_url" not in st.session_state:
    st.session_state.rag_api_url = default_api_url
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! Upload a document or ask me a question about the indexed "
                "knowledge base. I’ll answer with source citations."
            ),
            "is_greeting": True,
        }
    ]
if "rag_chat_session_id" not in st.session_state:
    st.session_state.rag_chat_session_id = uuid4().hex

client = RagApiClient(st.session_state.rag_api_url)
health: dict[str, Any] | None = None
backend_online = False
try:
    health = client.health()
    backend_online = str(health.get("status", "")).casefold() == "ok"
except RagApiError:
    pass

with st.sidebar:
    st.markdown('<div class="app-kicker">System</div>', unsafe_allow_html=True)
    st.header("RAG knowledge base")
    if backend_online:
        st.success("Backend and Qdrant ready")
    else:
        st.warning("Backend unavailable")

    with st.expander("Connection settings"):
        configured_url = st.text_input(
            "RAG API URL",
            value=st.session_state.rag_api_url,
        )
        if configured_url.rstrip("/") != st.session_state.rag_api_url.rstrip("/"):
            st.session_state.rag_api_url = configured_url
            st.rerun()
        if health:
            st.caption(f"Pipeline {health.get('pipeline_version', 'unknown')}")

    if st.button("New conversation", use_container_width=True):
        st.session_state.rag_messages = st.session_state.rag_messages[:1]
        st.session_state.rag_chat_session_id = uuid4().hex
        st.rerun()

    st.caption(
        "Documents are processed by the FastAPI backend and indexed in Qdrant."
    )

st.markdown('<div class="app-kicker">Your documents</div>', unsafe_allow_html=True)
st.title("RAG")
st.caption("Upload source material, then ask grounded questions about it.")

with st.container(border=True):
    st.subheader("Add a document")
    st.markdown(
        '<div class="upload-note">PDF, DOCX, PPTX, Markdown, text, CSV, or JSON</div>',
        unsafe_allow_html=True,
    )
    with st.form("upload_document", clear_on_submit=True):
        uploaded = st.file_uploader(
            "Choose a document",
            type=SUPPORTED_EXTENSIONS,
            disabled=not backend_online,
        )
        submitted = st.form_submit_button(
            "Upload and index",
            type="primary",
            disabled=not backend_online,
            use_container_width=True,
        )

    if submitted and uploaded is not None:
        try:
            with st.spinner("Uploading document…"):
                accepted = client.upload(
                    filename=uploaded.name,
                    content=uploaded.getvalue(),
                    media_type=uploaded.type,
                    title=None,
                )
            job = accepted.get("job", {})
            job_id = str(job.get("job_id") or "")
            if not job_id:
                raise RagApiError("The backend accepted the file without a job ID.")
            st.session_state.rag_job_id = job_id
            st.session_state.rag_uploaded_filename = uploaded.name
            st.success(str(accepted.get("message") or "Upload accepted."))
        except RagApiError as exc:
            st.error(str(exc))
    elif submitted:
        st.warning("Choose a document first.")

    job_id = st.session_state.get("rag_job_id")
    if job_id:
        try:
            _render_ingestion(client.ingestion(str(job_id)))
        except RagApiError as exc:
            st.error(str(exc))

if not backend_online:
    st.warning(
        f"The RAG API is unavailable at `{client.base_url}`. Start the backend to "
        "upload documents or ask questions."
    )

st.divider()
st.markdown('<div class="app-kicker">Ask your knowledge base</div>', unsafe_allow_html=True)
st.subheader("Chat")

for saved_message in st.session_state.rag_messages:
    _render_chat_message(saved_message)

question = st.chat_input(
    "Ask a question about your documents…",
    disabled=not backend_online,
    max_chars=10_000,
)
if question:
    user_message = {"role": "user", "content": question}
    st.session_state.rag_messages.append(user_message)
    _render_chat_message(user_message)

    try:
        with st.spinner("Retrieving evidence and preparing a grounded answer…"):
            response = client.chat(
                question=question,
                session_id=st.session_state.rag_chat_session_id,
            )
        assistant_message = {
            "role": "assistant",
            "content": response["answer"],
            "status": response.get("status", "answered"),
            "citations": response.get("citations", []),
            "confidence_score": response.get("confidence_score"),
            "timings": response.get("timings", {}),
            "diagnostics": {
                "model_id": response.get("model_id"),
                "embedding_model_id": response.get("embedding_model_id"),
                "usage": response.get("usage", {}),
                "timings": response.get("timings", {}),
                "debug": response.get("debug", {}),
            },
        }
    except RagApiError as exc:
        assistant_message = {
            "role": "assistant",
            "content": f"I couldn’t complete the request. {exc}",
            "status": "error",
            "citations": [],
        }

    st.session_state.rag_messages.append(assistant_message)
    _render_chat_message(assistant_message)
