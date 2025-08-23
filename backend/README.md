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

## Features

- Real-time CNN and RAG code generation
- Graph structure validation with error handling
- Support for pre-trained models (ResNet50, VGG16, MobileNetV2)
- Python code execution in isolated environment
- CORS enabled for frontend integration
