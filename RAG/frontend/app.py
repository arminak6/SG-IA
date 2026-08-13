from __future__ import annotations

import os

import streamlit as st
from api_client import RagApiClient, RagApiError

st.set_page_config(page_title="SG-IA RAG Lab", page_icon="R", layout="wide")
st.title("SG-IA grounded RAG lab")
st.caption(
    "Upload documents through FastAPI, inspect semantic retrieval, and ask grounded "
    "questions with source citations."
)

default_api_url = os.getenv("RAG_API_URL", "http://127.0.0.1:8001")
with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("RAG API URL", value=default_api_url)
    client = RagApiClient(api_url)
    if st.button("Check API", use_container_width=True):
        try:
            health = client.health()
            if health["status"] == "ok":
                st.success("API and Qdrant are ready")
            else:
                st.warning("API is running, but Qdrant is unavailable")
            st.json(health)
        except RagApiError as exc:
            st.error(str(exc))

upload_tab, documents_tab, retrieval_tab, chat_tab = st.tabs(
    ["Upload", "Documents", "Retrieval test", "Chat"]
)

with upload_tab:
    st.subheader("Add a document")
    st.write("Supported: PDF, DOCX, PPTX, Markdown, text, CSV, and JSON.")
    with st.form("upload_document"):
        uploaded = st.file_uploader(
            "Document", type=["pdf", "docx", "pptx", "md", "txt", "csv", "json"]
        )
        title = st.text_input("Display title (optional)")
        submitted = st.form_submit_button("Upload and index", type="primary")
    if submitted:
        if uploaded is None:
            st.warning("Choose a document first.")
        else:
            try:
                accepted = client.upload(
                    filename=uploaded.name,
                    content=uploaded.getvalue(),
                    media_type=uploaded.type,
                    title=title,
                )
                st.session_state["rag_job_id"] = accepted["job"]["job_id"]
                st.success(accepted["message"])
                st.json(accepted["job"])
            except RagApiError as exc:
                st.error(str(exc))

    st.divider()
    st.subheader("Ingestion status")
    job_id = st.text_input(
        "Job ID", value=st.session_state.get("rag_job_id", ""), key="job_id_input"
    )
    if st.button("Refresh status", disabled=not job_id):
        try:
            job = client.ingestion(job_id)
            status = job["status"]
            if status == "completed":
                st.success(f"Indexed {job.get('chunk_count', 0)} chunks")
            elif status == "failed":
                st.error(job.get("error") or "Ingestion failed")
            else:
                st.info(f"{status}: {job['stage']}")
            st.json(job)
        except RagApiError as exc:
            st.error(str(exc))

with documents_tab:
    st.subheader("Indexed documents")
    if st.button("Load documents", use_container_width=True):
        try:
            st.session_state["rag_documents"] = client.documents()
        except RagApiError as exc:
            st.error(str(exc))
    documents = st.session_state.get("rag_documents", [])
    if documents:
        st.dataframe(
            [
                {
                    "title": item["title"],
                    "file": item["filename"],
                    "pages": item.get("page_count"),
                    "elements": item["element_count"],
                    "chunks": item["chunk_count"],
                    "indexed": item["indexed_at"],
                    "document_id": item["document_id"],
                }
                for item in documents
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Load the document list after an ingestion finishes.")

with retrieval_tab:
    st.subheader("Inspect semantic retrieval")
    st.write(
        "This shows exactly which chunks Qdrant would provide to the future answer "
        "generator, including scores and source provenance."
    )
    documents = st.session_state.get("rag_documents", [])
    labels = {
        f"{item['title']} — {item['filename']}": item["document_id"]
        for item in documents
    }
    selected = st.multiselect(
        "Limit search to documents (optional)", options=list(labels)
    )
    with st.form("retrieval_form"):
        query = st.text_area("Question or search query", height=100)
        top_k = st.slider("Number of evidence chunks", 1, 20, 5)
        retrieve = st.form_submit_button("Search evidence", type="primary")
    if retrieve:
        if not query.strip():
            st.warning("Enter a question first.")
        else:
            try:
                result = client.search(
                    query=query,
                    top_k=top_k,
                    document_ids=[labels[label] for label in selected] or None,
                )
                st.metric("Retrieval latency", f"{result['latency_ms']:.0f} ms")
                if not result["hits"]:
                    st.warning("No matching evidence was found.")
                for rank, hit in enumerate(result["hits"], start=1):
                    pages = ", ".join(str(value) for value in hit["page_numbers"])
                    source = hit["filename"] + (f" · pages {pages}" if pages else "")
                    with st.expander(
                        f"#{rank} · score {hit['score']:.3f} · {source}",
                        expanded=rank <= 3,
                    ):
                        if hit["heading_path"]:
                            st.caption(" > ".join(hit["heading_path"]))
                        st.write(hit["text"])
                        st.json(hit["metadata"])
            except RagApiError as exc:
                st.error(str(exc))

with chat_tab:
    st.subheader("Ask the indexed knowledge base")
    st.write(
        "The backend retrieves semantic evidence, generates an answer constrained to "
        "that evidence, and returns the exact chunks cited by the model."
    )
    chat_documents = st.session_state.get("rag_documents", [])
    chat_labels = {
        f"{item['title']} — {item['filename']}": item["document_id"]
        for item in chat_documents
    }
    chat_selected = st.multiselect(
        "Limit the answer to documents (optional)",
        options=list(chat_labels),
        key="chat_document_filter",
    )
    with st.form("chat_form"):
        question = st.text_area("Question", height=120)
        chat_top_k = st.slider("Evidence chunks", 1, 20, 8)
        ask = st.form_submit_button("Ask RAG", type="primary")
    if ask:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Retrieving evidence and generating a grounded answer..."):
                    response = client.chat(
                        question=question,
                        top_k=chat_top_k,
                        document_ids=[chat_labels[label] for label in chat_selected]
                        or None,
                    )
                if response["status"] == "answered":
                    st.success("Grounded answer")
                else:
                    st.warning("Insufficient evidence")
                st.write(response["answer"])

                timings = response["timings"]
                metric_columns = st.columns(3)
                metric_columns[0].metric("Total", f"{timings['total_ms']:.0f} ms")
                metric_columns[1].metric(
                    "Retrieval", f"{timings['retrieval_ms']:.0f} ms"
                )
                metric_columns[2].metric(
                    "Generation", f"{timings['generation_ms']:.0f} ms"
                )

                st.markdown("#### Citations")
                if not response["citations"]:
                    st.caption("No evidence was cited.")
                for citation in response["citations"]:
                    pages = ", ".join(
                        str(value) for value in citation["page_numbers"]
                    )
                    label = citation["source_path"]
                    if pages:
                        label += f" · pages {pages}"
                    with st.expander(
                        f"{citation['evidence_id']} · {label} · score {citation['score']:.3f}"
                    ):
                        if citation["heading_path"]:
                            st.caption(" > ".join(citation["heading_path"]))
                        st.write(citation["excerpt"])
                        st.caption(f"Chunk: {citation['chunk_id']}")

                with st.expander("Debug metadata"):
                    st.json(
                        {
                            "model_id": response.get("model_id"),
                            "embedding_model_id": response["embedding_model_id"],
                            "usage": response["usage"],
                            "debug": response["debug"],
                        }
                    )
            except RagApiError as exc:
                st.error(str(exc))
