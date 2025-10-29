# DeepForge: Technical Project Summary

## Overview
DeepForge is a **full-stack no-code/low-code AI platform** that enables users to visually design machine learning pipelines and automatically generate production-ready Python code. It specializes in **CNN (Convolutional Neural Networks)** and **RAG (Retrieval-Augmented Generation)** systems through an intuitive drag-and-drop interface.

## Architecture

### Frontend Stack
- **React 19.1.1** - Modern React with concurrent features
- **Vite 7.1.2** - Lightning-fast build tool and dev server
- **ReactFlow 11.11.4** - Interactive node-based graph editor
- **Zustand 5.0.8** - Lightweight state management
- **React Router DOM 7.8.1** - Client-side routing with protected routes

### Backend Stack
- **Flask 3.0.3** - Lightweight Python web framework
- **PostgreSQL + Neon Cloud** - Managed database with psycopg2
- **JWT Authentication** - Token-based auth with bcrypt hashing
- **Flask-CORS** - Cross-origin resource sharing

### Database Schema
```sql
- users (authentication, profiles)
- subscriptions (pricing plans, features)
- user_subscriptions (billing relationships)
- chats/messages (conversation history)
```

## Core Features

### 1. Visual Pipeline Builders

#### CNN Builder
- **14 Component Types**: Dataset, Input Image, Conv2D, MaxPool, BatchNorm, Dropout, Flatten, Dense, Pre-trained Models, Data Augmentation
- **Pre-built Templates**: LeNet-style, Simple CNN, ResNet Transfer Learning, VGG-style Deep CNN, MobileNet Transfer
- **Transfer Learning**: ResNet50, VGG16, MobileNetV2, EfficientNet support
- **Real-time Validation**: Architecture validation with error/warning feedback

#### RAG Builder
- **8 Component Types**: Input Docs, Chunker, Embedder, Vector Store, Retriever, Reranker, LLM, Output
- **Vector Stores**: FAISS, Chroma, Milvus, PGVector support
- **LLM Integration**: GPT-4, GPT-3.5, Llama with temperature control
- **Enterprise Features**: S3 integration, reranking, multimodal support

### 2. Code Generation Engine
- **Real-time Generation**: 500ms debounced API calls for live updates
- **TensorFlow/Keras Output**: Production-ready Python code
- **Validation Pipeline**: Graph structure validation before generation
- **Error Handling**: Comprehensive feedback with actionable messages

### 3. Authentication & User Management
- **JWT-based Auth**: Secure token authentication
- **Protected Routes**: Frontend route guards
- **User Profiles**: Account management with subscriptions
- **Chat History**: Persistent conversation storage

## Technical Implementation

### Visual Graph Editor
```javascript
// ReactFlow integration for drag-and-drop pipeline building
const [nodes, setNodes, onNodesChange] = useNodesState(initial.nodes);
const [edges, setEdges, onEdgesChange] = useEdgesState(initial.edges);

// Real-time parameter updates
function updateNodeParam(nodeId, key, value, cast) {
  setNodes((nds) => nds.map((n) => {
    if (n.id !== nodeId) return n;
    const nextParams = { ...(n.data?.params || {}) };
    nextParams[key] = cast === 'number' ? Number(value) : value;
    return { ...n, data: { ...(n.data || {}), params: nextParams } };
  }));
}
```

### Code Generation Pipeline
```python
# Backend graph processing with topological ordering
def topological_order(nodes, edges):
    """Kahn's algorithm for dependency ordering"""
    
def validate_cnn_graph(nodes, edges):
    """Comprehensive CNN architecture validation"""
    
def generate_cnn_code(graph):
    """Convert visual graph to TensorFlow/Keras code"""
```

### API Architecture
```python
# RESTful endpoints with JWT middleware
@app.post('/generate')
@auth_required
def generate_code():
    # Graph validation → Code generation → Response
    
@app.post('/validate')
def validate_graph():
    # Structure validation → Error/warning collection
    
@app.post('/run')
def execute_code():
    # Sandboxed Python code execution
```

## Advanced Features

### 1. Schema-Driven Components
```javascript
// Extensible component system with parameter validation
const cnnSchemas = {
  conv2d: {
    title: 'Conv2D Layer',
    description: 'Learns visual features using sliding filters...',
    fields: [
      { key: 'filters', type: 'number', default: 32, help: 'Number of filters to learn' },
      { key: 'kernel', type: 'text', default: '3x3', help: 'Filter window size' }
    ]
  }
};
```

### 2. Graph Validation System
- **Topological Ordering**: Ensures proper layer sequencing
- **Component Compatibility**: Validates connections between layers
- **Parameter Validation**: Checks ranges and data types
- **Best Practices**: Enforces ML architecture guidelines
- **Error Categories**: Distinguishes between errors and warnings

### 3. Code Execution Sandbox
```python
# Isolated subprocess execution with timeout protection
def execute_code(code):
    result = subprocess.run([
        sys.executable, '-c', code
    ], capture_output=True, timeout=30, text=True)
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0
    }
```

### 4. Preset System
- **Pre-built Templates**: Rapid prototyping with proven architectures
- **One-click Deployment**: Instant pipeline creation
- **Custom Presets**: User-defined template creation

## Performance & Scalability

### Frontend Optimizations
- **Vite HMR**: Sub-second hot reloads
- **Code Splitting**: Route-based lazy loading
- **Debounced API**: Prevents excessive backend requests
- **Local Storage**: Offline-capable graph editing
- **React Concurrent**: Improved rendering performance

### Backend Optimizations
- **Connection Pooling**: Efficient database connections
- **JWT Stateless Auth**: Horizontally scalable authentication
- **Async Processing**: Non-blocking code generation
- **Error Boundaries**: Graceful failure handling

## Security & Production

### Security Measures
- **bcrypt Password Hashing**: Industry-standard security
- **JWT Token Expiration**: Configurable session timeouts
- **CORS Configuration**: Controlled cross-origin access
- **Input Validation**: Parameter sanitization
- **SQL Injection Prevention**: Parameterized queries

### Production Deployment
- **Gunicorn WSGI**: Production-grade Python serving
- **Environment Config**: `.env` based configuration
- **Health Checks**: Monitoring endpoints
- **Error Logging**: Comprehensive tracking

## Development Workflow

### Local Setup
```bash
# Frontend Development
npm install
npm run dev  # http://localhost:5173

# Backend Development
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py  # http://localhost:8000
```

### Code Quality Tools
- **ESLint 9.33.0**: Modern JavaScript/React linting
- **SWC Compiler**: Fast TypeScript/JavaScript compilation
- **Hot Reloading**: Instant development feedback

## File Structure
```
deepforge/
├── src/                          # React frontend
│   ├── components/
│   │   ├── builder/             # Visual builder components
│   │   ├── auth/                # Authentication components
│   │   ├── layouts/             # Layout components
│   │   └── sections/            # Landing page sections
│   ├── pages/                   # Route components
│   ├── api/                     # API client
│   ├── hooks/                   # Custom React hooks
│   └── styles/                  # CSS stylesheets
├── backend/                     # Flask backend
│   ├── app.py                   # Main Flask application
│   ├── requirements.txt         # Python dependencies
│   └── README.md               # Backend documentation
├── package.json                 # Frontend dependencies
├── vite.config.js              # Vite configuration
└── eslint.config.js            # ESLint configuration
```

## Key Components

### BaseBuilder.jsx
- Core visual builder component
- ReactFlow integration
- Real-time code generation
- Parameter editing interface
- Validation feedback display

### CNNBuilder.jsx & RAGBuilder.jsx
- Specialized builders for each ML domain
- Component palettes and schemas
- Pre-built architecture templates
- Domain-specific validation rules

### Flask Backend (app.py)
- RESTful API endpoints
- JWT authentication middleware
- Graph validation engine
- Code generation algorithms
- Sandboxed code execution

## Future Extensibility

The architecture supports easy extension for:
- **Additional ML Frameworks**: PyTorch, JAX, Scikit-learn
- **More Pipeline Types**: Computer Vision, NLP, Time Series
- **Cloud Integrations**: AWS, GCP, Azure deployment
- **Collaborative Features**: Real-time multi-user editing
- **Enterprise Features**: SSO, audit logs, team management

## Conclusion

DeepForge represents a modern, scalable approach to democratizing AI development through visual programming. It combines React's powerful ecosystem with Python's ML capabilities to create a production-ready platform that bridges the gap between no-code simplicity and professional ML development.

The project demonstrates advanced full-stack development practices, including real-time visual editing, sophisticated validation systems, secure authentication, and scalable architecture patterns suitable for enterprise deployment.