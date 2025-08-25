<<<<<<< HEAD
# DeepForge Backend

Flask-based backend server for the DeepForge AI builder application.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the server:
```bash
python app.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### Health Check
- **GET** `/health` - Returns server status

### Auth
- **POST** `/signup` – Body: `{username, email, password}` → creates user
- **POST** `/login` – Body: `{email, password}` → verifies user
  - Returns: `{ user, token }` where `token` is a Bearer JWT

Use JWT in requests:

```
Authorization: Bearer <token>
```

### Subscriptions
- **GET** `/subscriptions` – List subscription plans
- **POST** `/subscriptions` – Body: `{name, price, features}` → create plan (auth)
- **GET** `/subscriptions/{subscription_id}` – Get one plan
- **PUT** `/subscriptions/{subscription_id}` – Update plan (auth)
- **DELETE** `/subscriptions/{subscription_id}` – Delete plan (auth)
- **POST** `/users/{user_id}/subscriptions` – Body: `{subscription_id, end_date?}` → assign plan
- **GET** `/users/{user_id}/subscriptions` – List assigned plans for user (auth)

### Chats
- **POST** `/chats` – Create chat for current user (auth)
- **GET** `/users/{user_id}/chats` – List user's chats (auth, self only)

### Messages
- **POST** `/chats/{chat_id}/messages` – Body: `{sender: 'user'|'ai', message}` (auth, owned chat)
- **GET** `/chats/{chat_id}/messages` – List chat messages (auth, owned chat)

### Users
- **GET** `/users/me` – Get current user (auth)
- **PUT** `/users/me` – Update username and/or password (auth)
- **DELETE** `/users/me` – Delete current user (auth)

### Code Generation
- **POST** `/generate` - Generate Python code from graph data
  - Body: `{"graph": {...}, "builder_type": "cnn|rag"}`
  - Returns: `{"code": "...", "validation": {...}}`

### Code Execution
- **POST** `/run` - Execute Python code
  - Body: `{"code": "..."}`
  - Returns: `{"stdout": "...", "stderr": "...", "success": true|false}`

### Validation
- **POST** `/validate` - Validate graph structure
  - Body: `{"graph": {...}, "builder_type": "cnn|rag"}`
  - Returns: `{"validation": {"errors": [...], "warnings": [...]}}`

## Environment

Set Neon connection string via `.env` at `backend/.env`:

```
DATABASE_URL=postgresql://user:pass@host/db?sslmode=require
JWT_SECRET=change-this-secret
JWT_EXPIRES_MIN=60
```

## Features

- Real-time CNN and RAG code generation
- Graph structure validation with error handling
- Support for pre-trained models (ResNet50, VGG16, MobileNetV2)
- Python code execution in isolated environment
- CORS enabled for frontend integration
=======
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
>>>>>>> 421040b4e5ad063860aa0486547f7cb38f529574
