# Portable SG-IA deployment

The root deployment starts the complete comparison environment:

- Qdrant;
- RAG API and RAG Streamlit UI;
- LLM Wiki API and LLM Wiki Streamlit UI;
- the side-by-side comparison UI.

RAG and WIKI keep their independent Compose files for component development.
Use the root `compose.yaml` when running the complete application.

## Prerequisites

- Docker Desktop with Docker Compose;
- Python 3.11 or newer for the deployment helper scripts;
- Amazon Bedrock access for the configured embedding and generation models;
- a fresh temporary AWS credential file or environment credentials.

## Configure a computer

From the repository root:

```powershell
Copy-Item .env.example .env
Copy-Item WIKI/aws_credentials.example.json WIKI/aws_credentials.json
```

Put temporary credentials and a valid WIKI `bedrock_model_id` in
`WIKI/aws_credentials.json`, then set this in the root `.env`:

```dotenv
SGIA_AWS_CREDENTIALS_FILE=./WIKI/aws_credentials.json
```

Both files are ignored by Git. The credential file is mounted read-only and is
not copied into any image, corpus manifest, or state backup.

## Start everything

```powershell
.\deployment\start.ps1
```

The command creates the four named data/cache volumes if absent, builds and
starts all six services, waits for them, validates the base deployment,
idempotently bootstraps the private manifest into both RAG and WIKI, then
verifies credentials, API health, WIKI lint, the three UIs, and corpus
alignment. Existing matching source hashes are skipped. The root stack treats
the volumes as external so `down` cannot accidentally remove shared component
state.

| Surface | URL |
| --- | --- |
| Comparison UI | <http://localhost:8504> |
| RAG UI | <http://localhost:8502> |
| LLM Wiki UI | <http://localhost:8503> |
| RAG API docs | <http://localhost:8001/docs> |
| LLM Wiki API docs | <http://localhost:8002/docs> |
| Qdrant | <http://localhost:6337/dashboard> |

To start an intentionally empty installation without manifest bootstrap or
alignment validation:

```powershell
.\deployment\start.ps1 -SkipManifest
```

## Keep both knowledge bases aligned

`corpus-manifest.json` is a private, hash-verified declaration of the exact
source scope and the corresponding WIKI staging paths. It is intentionally
ignored because its filenames may be sensitive.

Generate it on the source computer from the currently ingested WIKI corpus:

```powershell
python deployment/make_corpus_manifest.py
```

After copying the private manifest and its selected files to another computer,
the normal `start.ps1` command performs bootstrap automatically. The same steps
can be run manually when developing or recovering a component:

```powershell
python deployment/bootstrap_knowledge.py
python deployment/validate_deployment.py
```

The bootstrap is repeatable: RAG skips source hashes already indexed, WIKI
skips unchanged sources, and differing WIKI staging files are never overwritten
unless `--replace-staged-wiki-source` is explicitly supplied. Bootstrap does
not delete old documents. Removing obsolete knowledge remains a reviewed,
approach-specific operation.

## Move existing indexed state

A plain folder copy cannot include Docker named volumes. Use the export/import
workflow when the existing RAG index must move without re-embedding.

On the source computer, stop the unified stack without deleting volumes, then
export:

```powershell
docker-compose -f compose.yaml stop
python deployment/export_state.py
docker-compose -f compose.yaml start
```

Copy the generated ignored directory under `deployment/backups/` separately
and securely. It contains private source and generated knowledge, but no AWS
credentials.

On a new computer with an otherwise empty clone:

```powershell
python deployment/import_state.py C:\path\to\backup --confirm-empty-restore
.\deployment\start.ps1
```

Import verifies every archive SHA-256, rejects unsafe archive paths, requires
stopped services, and refuses to overwrite non-empty RAG volumes, WIKI state,
material files, or a different manifest. It is deliberately not a merge tool.

## Useful commands

```powershell
# Re-run all readiness and alignment checks
python deployment/validate_deployment.py

# Inspect service state and logs
docker-compose -f compose.yaml ps
docker-compose -f compose.yaml logs --tail 100

# Stop while retaining indexed state
docker-compose -f compose.yaml stop

# Remove containers and the network while retaining named volumes
docker-compose -f compose.yaml down
```

The root stack treats its named volumes as external and does not delete them.
Do not use `down -v` from the independent component stacks unless permanent
deletion of that component's state has been explicitly reviewed and approved.
