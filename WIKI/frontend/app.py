"""Streamlit interface for the LLM Wiki project."""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st

try:
    from frontend.api_client import (
        ApiDocument,
        WikiApiClient,
        WikiApiError,
        build_manager_action_command,
    )
    from frontend.document_status import DocumentStatus, scan_documents
except ModuleNotFoundError:  # Also support `streamlit run app.py` from frontend/.
    from api_client import (
        ApiDocument,
        WikiApiClient,
        WikiApiError,
        build_manager_action_command,
    )
    from document_status import DocumentStatus, scan_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "backend" / "raw"
WIKI_DIR = PROJECT_ROOT / "backend" / "wiki"


st.set_page_config(
    page_title="LLM Wiki",
    page_icon="📚",
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
        .document-card {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 0.75rem;
            margin: 0.55rem 0;
            padding: 0.72rem 0.8rem;
        }
        .document-name {
            font-size: 0.9rem;
            font-weight: 650;
            line-height: 1.3;
            overflow-wrap: anywhere;
        }
        .document-meta {
            color: #64748b;
            font-size: 0.74rem;
            margin-top: 0.25rem;
        }
        .status-badge {
            border-radius: 999px;
            display: inline-block;
            font-size: 0.69rem;
            font-weight: 700;
            margin-top: 0.45rem;
            padding: 0.18rem 0.48rem;
        }
        .status-ingested {
            background: rgba(16, 185, 129, 0.14);
            color: #059669;
        }
        .status-pending {
            background: rgba(245, 158, 11, 0.16);
            color: #d97706;
        }
        .empty-state {
            border: 1px dashed rgba(148, 163, 184, 0.45);
            border-radius: 0.75rem;
            color: #64748b;
            font-size: 0.84rem;
            padding: 0.9rem;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


Document = DocumentStatus | ApiDocument

MANAGER_ACTIONS = {
    "fix": {
        "label": "Fix answer",
        "placeholder": (
            "It should say the approved limit is €500."
        ),
    },
    "update": {
        "label": "Update knowledge",
        "placeholder": (
            "The approved limit is now €600 from 2027; keep the existing eligibility rules."
        ),
    },
    "add": {
        "label": "Add knowledge",
        "placeholder": (
            "The support desk closes at 18:00."
        ),
    },
}


def format_size(size_bytes: int) -> str:
    """Format a file size for compact sidebar display."""

    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            precision = 0 if unit == "B" else 1
            return f"{size:.{precision}f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def render_document(document: Document, *, ingestion_enabled: bool) -> bool:
    """Render one source card and report a per-document ingestion request."""

    badge_class = "status-ingested" if document.is_ingested else "status-pending"
    modified = datetime.fromtimestamp(document.modified_timestamp).strftime("%d %b %Y")
    st.markdown(
        f"""
        <div class="document-card">
            <div class="document-name">📄 {escape(document.relative_path)}</div>
            <div class="document-meta">{format_size(document.size_bytes)} · {modified}</div>
            <span class="status-badge {badge_class}">{escape(document.status)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if document.is_ingested:
        return False
    return st.button(
        "Ingest this document",
        key=f"ingest-document::{document.relative_path}",
        use_container_width=True,
        disabled=not ingestion_enabled,
        help=(
            "Extract this document and update the wiki."
            if ingestion_enabled
            else "Start and configure the FastAPI/Bedrock backend first."
        ),
    )


def render_update_result(result: dict[str, Any]) -> None:
    processed = result.get("processed", [])
    skipped = result.get("skipped", [])
    failed = result.get("failed", [])
    expanded = bool(failed)

    with st.expander("Last wiki update", expanded=expanded):
        for item in processed:
            if isinstance(item, dict):
                detail = f" — {item['message']}" if item.get("message") else ""
                st.success(f"Ingested: {item.get('path', 'Unknown document')}{detail}")
            else:
                st.success(f"Ingested: {item}")
        for item in skipped:
            if isinstance(item, dict):
                detail = f" — {item['reason']}" if item.get("reason") else ""
                st.info(f"Skipped: {item.get('path', 'Unknown document')}{detail}")
            else:
                st.info(f"Skipped: {item}")
        for failure in failed:
            if isinstance(failure, dict):
                st.error(
                    f"Failed: {failure.get('path', 'Unknown document')} — "
                    f"{failure.get('error', 'Unknown error')}"
                )
        if not processed and not skipped and not failed:
            st.info("No documents needed an update.")


def render_sidebar(
    documents: list[Document],
    *,
    api: WikiApiClient,
    backend_online: bool,
    bedrock_ready: bool,
    health: dict[str, Any] | None,
    data_warning: str | None,
    wiki_page_count: int | None,
) -> None:
    with st.sidebar:
        st.markdown('<div class="app-kicker">Knowledge base</div>', unsafe_allow_html=True)
        st.header("Documents")

        if backend_online:
            configured = bool((health or {}).get("bedrock_configured", True))
            if configured:
                st.success("Backend online")
            else:
                st.warning("Backend online · Bedrock not configured")
        else:
            st.warning("Backend offline · showing local document status")
        if data_warning:
            st.caption(data_warning)

        ingested_count = sum(document.is_ingested for document in documents)
        pending_count = len(documents) - ingested_count
        pending_paths = [
            document.relative_path
            for document in documents
            if not document.is_ingested
        ]
        total_col, ingested_col, pending_col = st.columns(3)
        total_col.metric("Total", len(documents))
        ingested_col.metric("Ready", ingested_count)
        pending_col.metric("Pending", pending_count)
        if wiki_page_count is not None:
            st.caption(f"{wiki_page_count} wiki page(s) available")

        selected_pending_paths = st.multiselect(
            "Documents to ingest",
            options=pending_paths,
            default=[],
            placeholder="Select one or more pending documents",
            disabled=not pending_paths,
            help=(
                "PDF extraction can take several minutes. Start with one document "
                "when testing the pipeline."
            ),
        )
        if not backend_online:
            update_help = "Start the FastAPI backend to update the wiki."
        elif not bedrock_ready:
            update_help = "Configure AWS Bedrock before updating the wiki."
        elif not selected_pending_paths:
            update_help = "Select at least one pending document."
        else:
            update_help = "Ingest the selected documents with AWS Bedrock."

        refresh_col, update_col = st.columns(2)
        if refresh_col.button(
            "↻ Refresh",
            use_container_width=True,
            help="Refresh documents and wiki status.",
        ):
            st.session_state.pop("last_update_result", None)
            st.toast("Document list refreshed.")

        update_requested = update_col.button(
            "Update wiki",
            type="primary",
            use_container_width=True,
            disabled=not bedrock_ready or not selected_pending_paths,
            help=update_help,
        )

        if update_requested:
            pending_paths = selected_pending_paths
            try:
                with st.spinner(f"Ingesting {len(pending_paths)} document(s)…"):
                    result = api.update_wiki(pending_paths)
                st.session_state.last_update_result = result.as_dict()
                st.session_state.update_notice = "Wiki update finished."
                st.rerun()
            except WikiApiError as exc:
                st.error(str(exc))

        if notice := st.session_state.pop("update_notice", None):
            st.toast(notice)
        if result := st.session_state.get("last_update_result"):
            render_update_result(result)

        status_filter = st.radio(
            "Show",
            options=["All", "Pending", "Ingested"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )
        if status_filter == "Pending":
            visible_documents = [item for item in documents if not item.is_ingested]
        elif status_filter == "Ingested":
            visible_documents = [item for item in documents if item.is_ingested]
        else:
            visible_documents = documents

        if visible_documents:
            for document in visible_documents:
                if render_document(document, ingestion_enabled=bedrock_ready):
                    try:
                        with st.spinner("Ingesting 1 document..."):
                            result = api.update_wiki([document.relative_path])
                        st.session_state.last_update_result = result.as_dict()
                        st.session_state.update_notice = "Wiki update finished."
                        st.rerun()
                    except WikiApiError as exc:
                        st.error(str(exc))
        else:
            empty_message = (
                "No source documents found."
                if not documents
                else f"No {status_filter.lower()} documents."
            )
            st.markdown(
                f'<div class="empty-state">{escape(empty_message)}</div>',
                unsafe_allow_html=True,
            )


def render_chat_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        confidence_score = message.get("confidence_score")
        if isinstance(confidence_score, (int, float)) and not isinstance(
            confidence_score, bool
        ):
            status = str(message.get("status", "answered"))
            if status == "insufficient_knowledge":
                st.markdown(
                    f"**Abstention confidence: {float(confidence_score):.1f}/10**"
                )
            elif status in {"answered", "manager_action_applied"}:
                st.markdown(
                    f"**Evidence confidence: {float(confidence_score):.1f}/10**"
                )
        citations = message.get("citations", [])
        if citations:
            st.caption("Sources")
            for citation in citations:
                st.code(str(citation), language=None)


def render_manager_controls(messages: list[dict[str, Any]]) -> tuple[str | None, bool]:
    """Render explicit manager controls and return a queued command plus pending state."""

    latest_assistant = next(
        (message for message in reversed(messages) if message.get("role") == "assistant"),
        {},
    )
    manager_action = latest_assistant.get("manager_action")
    action_state = (
        str(manager_action.get("state", ""))
        if isinstance(manager_action, dict)
        else ""
    )
    pending = action_state in {"proposed", "needs_clarification", "failed", "error"}

    st.caption(
        "Manager maintenance is explicit. Normal chat stays in Q&A mode; knowledge "
        "changes always require a preview and confirmation."
    )

    if pending:
        if action_state == "needs_clarification":
            st.info(
                "This draft is not ready for approval. Provide the requested detail "
                "or cancel it before continuing Q&A."
            )
            with st.form("manager_action_clarification"):
                clarification = st.text_area(
                    "Clarification",
                    placeholder="Provide the missing detail requested above.",
                )
                clarify = st.form_submit_button("Send clarification", type="primary")
            if clarify:
                if clarification.strip():
                    return clarification.strip(), True
                st.warning("Enter the missing detail before sending.")
        else:
            st.info(
                "A manager action is ready. Confirm or cancel it before continuing Q&A."
            )

        confirm_col, cancel_col = st.columns(2)
        confirm = confirm_col.button(
            "Confirm action",
            type="primary",
            use_container_width=True,
            disabled=action_state == "needs_clarification",
        )
        cancel = cancel_col.button("Cancel action", use_container_width=True)
        if confirm:
            return "/confirm", True
        if cancel:
            return "/cancel", True
        return None, True

    has_prior_question = any(message.get("role") == "user" for message in messages)
    fix_col, update_col, add_col = st.columns(3)
    if fix_col.button(
        "Fix answer",
        use_container_width=True,
        disabled=not has_prior_question,
        help="Correct the previous answer using knowledge already present in the Wiki.",
    ):
        if st.session_state.get("manager_action_mode") != "fix":
            st.session_state.pop("manager_action_details", None)
        st.session_state.manager_action_mode = "fix"
    if update_col.button(
        "Update knowledge",
        use_container_width=True,
        help="Replace outdated or incorrect maintained knowledge.",
    ):
        if st.session_state.get("manager_action_mode") != "update":
            st.session_state.pop("manager_action_details", None)
        st.session_state.manager_action_mode = "update"
    if add_col.button(
        "Add knowledge",
        use_container_width=True,
        help="Add genuinely new manager-approved knowledge.",
    ):
        if st.session_state.get("manager_action_mode") != "add":
            st.session_state.pop("manager_action_details", None)
        st.session_state.manager_action_mode = "add"

    mode = st.session_state.get("manager_action_mode")
    config = MANAGER_ACTIONS.get(mode)
    if config is None:
        return None, False

    st.info(
        f"{config['label']} mode — describe the factual change in a short sentence. "
        "The preview will show the complete proposed knowledge. "
        f"Do not type `/{mode}`; the interface adds it automatically."
    )
    with st.form("manager_action_form"):
        st.markdown(f"**{config['label']}**")
        details = st.text_area(
            "What changed?",
            placeholder=config["placeholder"],
            key="manager_action_details",
        )
        preview = st.form_submit_button("Preview action", type="primary")
    if st.button("Close manager action", key="close_manager_action"):
        st.session_state.clear_manager_action_on_next_run = True
        st.rerun()
    if preview:
        try:
            return build_manager_action_command(str(mode), details), True
        except ValueError as exc:
            st.warning(str(exc))
    return None, True


api = WikiApiClient()
health: dict[str, Any] | None = None
backend_online = False
data_warning: str | None = None

try:
    health = api.health()
    backend_online = True
except WikiApiError:
    pass

bedrock_ready = backend_online and bool((health or {}).get("bedrock_configured", False))

local_documents = scan_documents(RAW_DIR, WIKI_DIR)
documents: list[Document] = local_documents
wiki_page_count: int | None = None

if backend_online:
    try:
        documents = api.list_documents()
    except WikiApiError:
        data_warning = "Could not load API documents; using a local scan."
    try:
        wiki_page_count = len(api.list_pages())
    except WikiApiError:
        wiki_page_count = None

render_sidebar(
    documents,
    api=api,
    backend_online=backend_online,
    bedrock_ready=bedrock_ready,
    health=health,
    data_warning=data_warning,
    wiki_page_count=wiki_page_count,
)

st.markdown('<div class="app-kicker">Ask your knowledge base</div>', unsafe_allow_html=True)
st.title("LLM Wiki")
st.caption("Ask grounded questions about the Markdown wiki maintained from your sources.")

local_wiki_has_pages = WIKI_DIR.is_dir() and any(
    page.relative_to(WIKI_DIR).as_posix().casefold() not in {"index.md", "log.md"}
    for page in WIKI_DIR.rglob("*.md")
    if page.is_file()
)
wiki_has_pages = wiki_page_count > 0 if wiki_page_count is not None else local_wiki_has_pages
if not wiki_has_pages:
    st.info(
        "Your wiki does not contain any Markdown pages yet. Start the backend, then "
        "use **Update wiki** to ingest pending documents."
    )

if not backend_online:
    st.warning(
        f"The Q&A API is offline at `{api.base_url}`. Start the FastAPI backend to "
        "update the wiki or ask questions."
    )
elif not bedrock_ready:
    st.warning(
        "The backend is online, but AWS Bedrock is not configured. Check the "
        "server configuration before updating the wiki or asking questions."
    )

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! Ask me a question and I’ll answer from the maintained wiki "
                "with citations."
            ),
            "citations": [],
        }
    ]

if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = uuid4().hex

if st.session_state.pop("clear_manager_action_on_next_run", False):
    st.session_state.pop("manager_action_mode", None)
    st.session_state.pop("manager_action_details", None)

for message in st.session_state.messages:
    render_chat_message(message)

manager_message, manager_pending = render_manager_controls(st.session_state.messages)
chat_question = st.chat_input(
    "Ask a question — use the buttons above to change knowledge.",
    disabled=manager_pending,
)
question = manager_message or chat_question

if question:
    user_message = {"role": "user", "content": question, "citations": []}
    st.session_state.messages.append(user_message)
    render_chat_message(user_message)

    if not backend_online:
        assistant_message = {
            "role": "assistant",
            "content": (
                "I can’t answer yet because the backend is offline. Start it with "
                "`uvicorn backend.main:app --reload`, then try again."
            ),
            "citations": [],
        }
    elif not bedrock_ready:
        assistant_message = {
            "role": "assistant",
            "content": (
                "I can’t answer yet because AWS Bedrock is not configured on the "
                "backend. Check `aws_credentials.json` or the AWS environment settings."
            ),
            "citations": [],
        }
    else:
        try:
            with st.spinner("Working with the wiki…"):
                response = api.chat(
                    question,
                    session_id=st.session_state.chat_session_id,
                )
            assistant_message = {
                "role": "assistant",
                "content": response.answer,
                "citations": list(response.citations),
                "confidence_score": response.confidence_score,
                "status": response.status,
                "manager_action": response.manager_action,
            }
            if manager_message is not None and response.status != "manager_action_error":
                st.session_state.clear_manager_action_on_next_run = True
        except WikiApiError as exc:
            assistant_message = {
                "role": "assistant",
                "content": f"I couldn’t complete that request. {exc}",
                "citations": [],
            }

    st.session_state.messages.append(assistant_message)
    st.rerun()
