# LLM Wiki frontend

This directory contains the Streamlit interface for the project. It lists raw
documents and their ingestion status, triggers wiki updates through FastAPI,
and sends grounded questions to the backend chat endpoint.

Select one or more pending documents before choosing **Update wiki**. Start
with one document when testing because structured PDF extraction can take
several minutes before the Bedrock wiki-maintenance step begins.

## Run locally

From the project root, install the frontend dependencies:

```powershell
python -m pip install -r frontend/requirements.txt
```

Start the backend in one terminal:

```powershell
uvicorn backend.main:app --reload
```

Then start Streamlit in another terminal:

```powershell
streamlit run frontend/app.py
```

The frontend connects to `http://127.0.0.1:8000` by default. To use a different
backend address, set `LLM_WIKI_API_URL` before starting Streamlit:

```powershell
$env:LLM_WIKI_API_URL = "http://127.0.0.1:9000"
streamlit run frontend/app.py
```

When the API is unavailable, the sidebar remains usable and recursively scans
`backend/raw`. Document status is inferred from source references in
`backend/wiki`, while update and Q&A actions clearly remain unavailable until
the backend is online.
