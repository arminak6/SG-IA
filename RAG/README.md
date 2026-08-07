# RAG implementation

This directory will contain the standard Retrieval-Augmented Generation side
of the SG-IA comparison.

The RAG system must remain independently runnable and should provide:

- a FastAPI backend for ingestion, retrieval, generation, and citations;
- a Streamlit frontend that calls the backend API;
- structured answer, source, timing, usage, and debug metadata;
- tests that do not require paid model calls by default;
- the same approved source scope and evaluation questions used by `WIKI/`.

Planned layout:

```text
RAG/
|-- backend/
|   |-- tests/
|   `-- main.py
`-- frontend/
    |-- tests/
    `-- app.py
```

Model provider, embedding model, vector database, and chunking strategy have
not yet been selected. Those decisions should be made explicitly before the
first RAG implementation commit.

