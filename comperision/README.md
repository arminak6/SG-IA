# SG-IA comparison chatbot

This Streamlit interface sends one question to the RAG and LLM Wiki FastAPI
backends concurrently. It shows the LLM Wiki answer on the left and the RAG
answer on the right, including status, citations, latency, model information,
usage, and retrieval/navigation diagnostics.

The interface is deliberately an API client. It does not read Qdrant, Wiki
Markdown, or source documents, and it performs no ingestion. Start and ingest
the two independent systems first so both contain the same approved source
scope.

## Docker quick start

Start the existing stacks in separate terminals:

```powershell
cd C:\sinergia_ak\chatbot\SG-IA\RAG
docker-compose up -d
```

```powershell
cd C:\sinergia_ak\chatbot\SG-IA\WIKI
docker-compose up -d
```

Then start the comparison UI:

```powershell
cd C:\sinergia_ak\chatbot\SG-IA\comperision
docker-compose up -d --build
```

Open <http://localhost:8504>. The container defaults to the host-published RAG
API at port 8001 and WIKI API at port 8002. Copy `.env.example` to `.env` only
when those URLs or the UI port need to be changed.

The stacks remain independent: if one backend is unavailable, its column shows
the error while the other backend's completed answer remains visible.
The comparison request always asks RAG for its configured maximum of 10 final
reranked evidence chunks; this is intentionally not exposed as a UI control.

## Local development

```powershell
cd C:\sinergia_ak\chatbot\SG-IA\comperision
python -m pip install -r requirements.txt
streamlit run app.py
```

Local defaults are `http://127.0.0.1:8001` for RAG and
`http://127.0.0.1:8002` for WIKI. They can be changed in the sidebar or with
the `RAG_API_URL` and `WIKI_API_URL` environment variables.

## Offline tests

```powershell
python -m unittest discover -s tests -v
```

The tests use fake HTTP sessions and do not call AWS, Qdrant, or either live
backend.
