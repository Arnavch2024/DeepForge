# DeepForge Backend (Flask)

## Setup
1. Create and activate a virtual environment
   - Windows (PowerShell):
     - `python -m venv .venv`
     - `.venv\Scripts\Activate.ps1`
   - macOS/Linux:
     - `python3 -m venv .venv`
     - `source .venv/bin/activate`
2. Install dependencies
   - `pip install -r backend/requirements.txt`

## Run
- `python backend/app.py`
- Server runs at `http://localhost:8000`
- Health: `GET /health`
- Generate code: `POST /generate` with JSON `{ type: 'cnn'|'rag', graph: { nodes:[], edges:[] } }`
- Run code: `POST /run` with JSON `{ code: 'python_source_code' }`

## Notes
- Code generation is placeholder; wire your graph-to-code logic inside `generate_cnn_code` / `generate_rag_code`.
- Execution runs in a subprocess with a timeout; expand with sandboxing as needed. 