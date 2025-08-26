from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import sys
import tempfile
import textwrap
import json
import os
import traceback

from typing import Dict, List, Tuple, Any, Optional

import psycopg2
import bcrypt
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import jwt
from datetime import datetime, timedelta, timezone
from psycopg2 import errors

load_dotenv()
app = Flask(__name__)
CORS(app)

# -------------------------------
# Database setup (Neon Postgres)
# -------------------------------

DB_URL = os.getenv('DATABASE_URL') or os.getenv('NEON_DB_URL') or 'postgresql://neondb_owner:npg_r3WRIEaLX0Ok@ep-holy-art-a1ny5k2k-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require'
JWT_SECRET = os.getenv('JWT_SECRET') or 'change-this-secret'
JWT_EXPIRES_MIN = int(os.getenv('JWT_EXPIRES_MIN') or '60')

def get_db_connection():
	conn = psycopg2.connect(DB_URL)
	return conn

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id BIGSERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS subscriptions (
    subscription_id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    price DECIMAL(10,2) NOT NULL DEFAULT 0,
    features TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_subscriptions (
    user_subscription_id SERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(user_id) ON DELETE CASCADE,
    subscription_id INT REFERENCES subscriptions(subscription_id),
    start_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_date TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS chats (
    chat_id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(user_id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    message_id BIGSERIAL PRIMARY KEY,
    chat_id BIGINT REFERENCES chats(chat_id) ON DELETE CASCADE,
    sender VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Flat chat messages table for simple history view (optional alongside chats/messages)
CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(user_id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def init_db_schema() -> None:
	try:
		with get_db_connection() as conn:
			with conn.cursor() as cur:
				cur.execute(SCHEMA_SQL)
				conn.commit()
	except Exception as e:
		print(f"[DB] Schema init failed: {e}")
		raise


# -------------------------------
# Auth helpers (JWT)
# -------------------------------

def create_access_token(user: Dict[str, any]) -> str:
	payload = {
		"sub": str(user["user_id"]),
		"username": user.get("username"),
		"email": user.get("email"),
		"iat": datetime.now(timezone.utc),
		"exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRES_MIN)
	}
	return jwt.encode(payload, JWT_SECRET, algorithm='HS256')


def _get_bearer_token() -> Optional[str]:
	auth = request.headers.get('Authorization') or ''
	if auth.lower().startswith('bearer '):
		return auth.split(' ', 1)[1].strip()
	return None


def get_current_user() -> Optional[Dict[str, any]]:
	token = _get_bearer_token()
	if not token:
		return None
	try:
		payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
		user_id = int(payload.get('sub'))
		with get_db_connection() as conn:
			with conn.cursor(cursor_factory=RealDictCursor) as cur:
				cur.execute("SELECT user_id, username, email, created_at FROM users WHERE user_id = %s", (user_id,))
				return cur.fetchone()
	except Exception:
		return None


def auth_required(fn):
	def wrapper(*args, **kwargs):
		user = get_current_user()
		if not user:
			return jsonify({"error": "Unauthorized"}), 401
		request.user = user
		return fn(*args, **kwargs)
	# Preserve function name for Flask
	wrapper.__name__ = fn.__name__
	return wrapper

# Simple health check
@app.get('/health')
def health():
	return jsonify({"status": "ok"})


# -------------------------------
# Validation and Error Handling
# -------------------------------

class ValidationError(Exception):
	"""Custom exception for validation errors"""
	pass

def validate_cnn_graph(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Validate CNN graph structure and return dictionary of errors and warnings.
    
    Args:
        nodes: List of node objects in the graph
        edges: List of edge objects connecting nodes
        
    Returns:
        Dict containing 'errors' and 'warnings' lists
    """
    errors = []
    warnings = []
    
    if not nodes:
        errors.append("No nodes found in the graph")
        return {"errors": errors, "warnings": warnings}
    
    # Check for required nodes
    node_types = [_n_type(n) for n in nodes]
    if 'dataset' not in node_types:
        errors.append("CNN must have a Dataset component")
    if 'inputImage' not in node_types:
        errors.append("CNN must have exactly one input layer")
    if 'output' not in node_types:
        errors.append("CNN must have exactly one output layer")
    
    # Check for multiple input/output nodes
    input_count = node_types.count('inputImage')
    output_count = node_types.count('output')
    if input_count > 1:
        errors.append(f"CNN cannot have multiple input layers (found {input_count})")
    if output_count > 1:
        errors.append(f"CNN cannot have multiple output layers (found {output_count})")
    
    # Check for isolated nodes (no connections)
    connected_nodes = set()
    for edge in edges:
        connected_nodes.add(edge.get('source'))
        connected_nodes.add(edge.get('target'))
    
    for node in nodes:
        if node['id'] not in connected_nodes:
            warnings.append(f"Node '{_n_label(node)}' ({_n_type(node)}) is not connected to the graph")
    
    # Check for cycles (basic check)
    if len(edges) >= len(nodes):
        warnings.append("Graph may contain cycles - ensure proper layer ordering")
    
    # Enhanced layer compatibility and connection validation
    conv_layers = [n for n in nodes if _n_type(n) in ['conv2d', 'maxpool']]
    dense_layers = [n for n in nodes if _n_type(n) in ['dense', 'output']]
    
    # Check layer ordering and compatibility
    if conv_layers and dense_layers:
        # Check if flatten exists between conv and dense
        has_flatten = any(_n_type(n) == 'flatten' for n in nodes)
        if not has_flatten:
            errors.append("Cannot connect convolutional layers directly to dense layers - add a Flatten layer in between")
	
    # Check for invalid layer sequences
    for i, node in enumerate(nodes):
        node_type = _n_type(node)
        
        # Check what comes after this node
        for edge in edges:
            if edge.get('source') == node['id']:
                target_node = next((n for n in nodes if n['id'] == edge.get('target')), None)
                if target_node:
                    target_type = _n_type(target_node)
                    
                    # Invalid sequences
                    if node_type == 'flatten' and target_type in ['conv2d', 'maxpool']:
                        errors.append(f"Cannot connect Flatten layer to {target_type} - Flatten should only connect to Dense/Output layers")
                    
                    if node_type == 'output' and target_type != 'output':
                        errors.append(f"Output layer cannot have outgoing connections to {target_type}")
                    
                    if node_type == 'inputImage' and target_type not in ['conv2d', 'maxpool', 'batchnorm']:
                        errors.append(f"Input layer should connect to Conv2D, MaxPool, or BatchNorm, not {target_type}")
                    
                    if node_type == 'batchnorm' and target_type == 'batchnorm':
                        warnings.append("Consecutive BatchNorm layers are usually unnecessary - consider removing one")
                    
                    if node_type == 'dropout' and target_type == 'dropout':
                        warnings.append("Consecutive Dropout layers may cause excessive regularization")
                    
                    if node_type == 'maxpool' and target_type == 'maxpool':
                        warnings.append("Consecutive MaxPool layers will rapidly reduce spatial dimensions - consider alternatives")
	
    # Parameter validation and best practices
    for node in nodes:
        node_type = _n_type(node)
        params = _n_params(node)
        
        if node_type == 'inputImage':
            width = params.get('width', 0)
            height = params.get('height', 0)
            channels = params.get('channels', 0)
            
            if width <= 0 or height <= 0:
                errors.append(f"Input dimensions must be positive (got {width}x{height})")
            if channels not in [1, 3]:
                errors.append(f"Input channels should be 1 (grayscale) or 3 (RGB), got {channels}")
            if width > 1024 or height > 1024:
                warnings.append(f"Large input dimensions ({width}x{height}) may cause memory issues")
		
        elif node_type == 'conv2d':
            filters = params.get('filters', 0)
            kernel = params.get('kernel', '3x3')
            
            if filters <= 0:
                errors.append(f"Conv2D filters must be positive, got {filters}")
            if filters > 512:
                warnings.append(f"Very large number of filters ({filters}) may cause overfitting")
            
            # Validate kernel size
            if isinstance(kernel, str):
                if 'x' in kernel:
                    parts = kernel.split('x')
                    if len(parts) == 2:
                        try:
                            k_w, k_h = int(parts[0]), int(parts[1])
                            if k_w <= 0 or k_h <= 0:
                                errors.append(f"Invalid kernel size: {kernel}")
                            if k_w > 11 or k_h > 11:
                                warnings.append(f"Very large kernel size ({kernel}) may be inefficient")
                        except ValueError:
                            errors.append(f"Invalid kernel size format: {kernel}")
		
        elif node_type == 'dense':
            units = params.get('units', 0)
            if units <= 0:
                errors.append(f"Dense layer units must be positive, got {units}")
            if units > 4096:
                warnings.append(f"Very large dense layer ({units} units) may cause overfitting")
		
        elif node_type == 'output':
            classes = params.get('classes', 0)
            if classes <= 0:
                errors.append(f"Output classes must be positive, got {classes}")
            if classes > 1000:
                warnings.append(f"Very large number of classes ({classes}) may require more training data")
        elif node_type == 'dataset':
            train_split = params.get('train_split', 0)
            validation_split = params.get('validation_split', 0)
            test_split = params.get('test_split', 0)
            if train_split + validation_split + test_split != 100:
                errors.append("The sum of train, validation, and test splits must be 100")
            if params.get('batch_size', 0) <= 0:
                errors.append("Batch size must be a positive number")
        elif node_type == 'randomFlip':
            mode = params.get('mode', 'horizontal')
            if mode not in ['horizontal', 'vertical', 'horizontal_and_vertical']:
                errors.append(f"Invalid flip mode: {mode}")
        elif node_type == 'randomRotation':
            factor = params.get('factor', 0.0)
            if not isinstance(factor, (int, float)):
                errors.append("Rotation factor must be a number")
        elif node_type == 'randomZoom':
            height_factor = params.get('height_factor', 0.0)
            width_factor = params.get('width_factor', 0.0)
            if not isinstance(height_factor, (int, float)) or not isinstance(width_factor, (int, float)):
                errors.append("Zoom factors must be numbers")
        elif node_type == 'randomContrast':
            factor = params.get('factor', 0.0)
            if not isinstance(factor, (int, float)):
                errors.append("Contrast factor must be a number")
	
    # Check for reasonable layer counts
    if len(conv_layers) > 10:
        warnings.append("Very deep CNN detected - consider reducing layers for training efficiency")
    if len(dense_layers) > 5:
        warnings.append("Many dense layers detected - consider regularization to prevent overfitting")
    
    # Check for missing essential layers
    if conv_layers and not any(_n_type(n) == 'maxpool' for n in nodes):
        warnings.append("Consider adding MaxPool layers after Conv2D to reduce spatial dimensions")
    
    # Check for pre-trained model compatibility
    pretrained_nodes = [n for n in nodes if _n_type(n) == 'pretrained']
    if pretrained_nodes:
        if len(pretrained_nodes) > 1:
            errors.append("Cannot have multiple pre-trained models in one CNN")
        
        # Check if pre-trained model is properly positioned
        for pretrained_node in pretrained_nodes:
            has_input_before = False
            has_dense_after = False
            
            for edge in edges:
                if edge.get('target') == pretrained_node['id']:
                    source_node = next((n for n in nodes if n['id'] == edge.get('source')), None)
                    if source_node and _n_type(source_node) == 'inputImage':
                        has_input_before = True
                
                if edge.get('source') == pretrained_node['id']:
                    target_node = next((n for n in nodes if n['id'] == edge.get('target')), None)
                    if target_node and _n_type(target_node) in ['dense', 'output']:
                        has_dense_after = True
            
            if not has_input_before:
                errors.append("Pre-trained model must come after Input Image layer")
            if not has_dense_after:
                errors.append("Pre-trained model must connect to Dense or Output layers")
    
    return errors + warnings

def validate_rag_graph(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[str]:
	"""Validate RAG graph structure and return list of errors/warnings"""
	errors = []
	warnings = []
	
	if not nodes:
		errors.append("No nodes found in the graph")
		return errors
	
	# Check for required nodes
	node_types = [_n_type(n) for n in nodes]
	required = ['inputDocs', 'embed', 'chunk', 'vectorstore', 'retriever', 'llm']
	missing = [req for req in required if req not in node_types]
	if missing:
		errors.append(f"RAG pipeline missing required components: {', '.join(missing)}")
	
	# Check for multiple instances of single components
	single_components = ['inputDocs', 'embed', 'chunk', 'vectorstore']
	for comp in single_components:
		count = node_types.count(comp)
		if count > 1:
			errors.append(f"RAG can only have one {comp} component (found {count})")
	
	# Check for isolated nodes
	connected_nodes = set()
	for edge in edges:
		connected_nodes.add(edge.get('source'))
		connected_nodes.add(edge.get('target'))
	
	for node in nodes:
		if node['id'] not in connected_nodes:
			warnings.append(f"Node '{_n_label(node)}' ({_n_type(node)}) is not connected to the pipeline")
	
	# Enhanced component compatibility and connection validation
	for i, node in enumerate(nodes):
		node_type = _n_type(node)
		
		# Check what comes after this node
		for edge in edges:
			if edge.get('source') == node['id']:
				target_node = next((n for n in nodes if n['id'] == edge.get('target')), None)
				if target_node:
					target_type = _n_type(target_node)
					
					# Invalid sequences
					if node_type == 'inputDocs' and target_type not in ['chunk']:
						errors.append(f"Input Docs should connect to Chunker, not {target_type}")
					
					if node_type == 'chunk' and target_type not in ['embed']:
						errors.append(f"Chunker should connect to Embedder, not {target_type}")
					
					if node_type == 'embed' and target_type not in ['vectorstore']:
						errors.append(f"Embedder should connect to Vector Store, not {target_type}")
					
					if node_type == 'vectorstore' and target_type not in ['retriever']:
						errors.append(f"Vector Store should connect to Retriever, not {target_type}")
					
					if node_type == 'retriever' and target_type not in ['reranker', 'llm']:
						errors.append(f"Retriever should connect to Reranker or LLM, not {target_type}")
					
					if node_type == 'reranker' and target_type not in ['llm']:
						errors.append(f"Reranker should connect to LLM, not {target_type}")
					
					if node_type == 'llm' and target_type not in ['output']:
						errors.append(f"LLM should connect to Output, not {target_type}")
					
					if node_type == 'output' and target_type != 'output':
						errors.append(f"Output component cannot have outgoing connections to {target_type}")
	
	# Check for proper pipeline flow
	has_chunk_to_embed = False
	has_embed_to_store = False
	has_store_to_retriever = False
	has_retriever_to_llm = False
	
	for edge in edges:
		source_node = next((n for n in nodes if n['id'] == edge.get('source')), None)
		target_node = next((n for n in nodes if n['id'] == edge.get('target')), None)
		
		if source_node and target_node:
			source_type = _n_type(source_node)
			target_type = _n_type(target_node)
			
			if source_type == 'chunk' and target_type == 'embed':
				has_chunk_to_embed = True
			if source_type == 'embed' and target_type == 'vectorstore':
				has_embed_to_store = True
			if source_type == 'vectorstore' and target_type == 'retriever':
				has_store_to_retriever = True
			if source_type == 'retriever' and target_type == 'llm':
				has_retriever_to_llm = True
	
	# Validate essential connections
	if not has_chunk_to_embed:
		errors.append("Chunker must connect to Embedder for document processing")
	if not has_embed_to_store:
		errors.append("Embedder must connect to Vector Store for storage")
	if not has_store_to_retriever:
		errors.append("Vector Store must connect to Retriever for search")
	if not has_retriever_to_llm:
		errors.append("Retriever must connect to LLM for answer generation")
	
	# Check for reranker placement
	reranker_nodes = [n for n in nodes if _n_type(n) == 'reranker']
	if reranker_nodes:
		if len(reranker_nodes) > 1:
			errors.append("Cannot have multiple rerankers in one pipeline")
		
		# Check if reranker is properly placed between retriever and LLM
		for reranker_node in reranker_nodes:
			has_retriever_before = False
			has_llm_after = False
			
			for edge in edges:
				if edge.get('target') == reranker_node['id']:
					source_node = next((n for n in nodes if n['id'] == edge.get('source')), None)
					if source_node and _n_type(source_node) == 'retriever':
						has_retriever_before = True
				
				if edge.get('source') == reranker_node['id']:
					target_node = next((n for n in nodes if n['id'] == edge.get('target')), None)
					if target_node and _n_type(target_node) == 'llm':
						has_llm_after = True
			
			if not has_retriever_before:
				errors.append("Reranker must come after Retriever")
			if not has_llm_after:
				errors.append("Reranker must connect to LLM")
	
	# Check for reasonable component counts
	if len(nodes) > 15:
		warnings.append("Very complex RAG pipeline detected - consider simplifying for maintainability")
	
	# Check for missing essential components
	if 'reranker' not in node_types:
		warnings.append("Consider adding a Reranker for improved retrieval quality")
	
	# Parameter validation and best practices
	for node in nodes:
		node_type = _n_type(node)
		params = _n_params(node)
		
		if node_type == 'chunk':
			chunk_size = params.get('chunkSize', 0)
			overlap = params.get('overlap', 0)
			
			if chunk_size <= 0:
				errors.append(f"Chunk size must be positive, got {chunk_size}")
			if chunk_size > 2000:
				warnings.append(f"Very large chunk size ({chunk_size}) may reduce retrieval precision")
			if overlap < 0:
				errors.append(f"Overlap cannot be negative, got {overlap}")
			if overlap >= chunk_size:
				errors.append(f"Overlap ({overlap}) must be less than chunk size ({chunk_size})")
			if overlap > chunk_size * 0.5:
				warnings.append(f"High overlap ({overlap}) may create redundant chunks")
		
		elif node_type == 'embed':
			dim = params.get('dim', 0)
			if dim <= 0:
				errors.append(f"Embedding dimension must be positive, got {dim}")
			if dim > 4096:
				warnings.append(f"Very high embedding dimension ({dim}) may cause performance issues")
		
		elif node_type == 'retriever':
			top_k = params.get('topK', 0)
			if top_k <= 0:
				errors.append(f"Top-K must be positive, got {top_k}")
			if top_k > 50:
				warnings.append(f"Very high Top-K ({top_k}) may slow down the pipeline")
		
		elif node_type == 'llm':
			temperature = params.get('temperature', 0.7)
			if temperature < 0 or temperature > 2:
				errors.append(f"Temperature must be between 0 and 2, got {temperature}")
			if temperature > 1.5:
				warnings.append(f"Very high temperature ({temperature}) may produce unreliable answers")
	
	return errors + warnings

def validate_graph_structure(builder_type: str, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""Validate graph structure and return validation results"""
	if builder_type == 'cnn':
		issues = validate_cnn_graph(nodes, edges)
	elif builder_type == 'rag':
		issues = validate_rag_graph(nodes, edges)
	else:
		return {"valid": False, "errors": ["Invalid builder type"], "warnings": []}
	
	errors = [issue for issue in issues if "must" in issue.lower() or "cannot" in issue.lower()]
	warnings = [issue for issue in issues if issue not in errors]
	
	return {
		"valid": len(errors) == 0,
		"errors": errors,
		"warnings": warnings
	}


# -------------------------------
# Utilities for graph processing
# -------------------------------

def _to_tuple_2(val: str, default: Tuple[int, int] = (1, 1)) -> Tuple[int, int]:
	"""Parse 'AxB' or 'A,B' or 'A A' into a 2-tuple of ints."""
	if not val:
		return default
	s = str(val).lower().replace(' ', 'x').replace(',', 'x')
	parts = [p for p in s.split('x') if p]
	if len(parts) == 1:
		try:
			v = int(parts[0])
			return (v, v)
		except Exception:
			return default
	try:
		return (int(parts[0]), int(parts[1]))
	except Exception:
		return default


def _n_type(n: Dict[str, Any]) -> str:
	data = n.get('data') or {}
	return data.get('type') or n.get('type') or ''


def _n_label(n: Dict[str, Any]) -> str:
	data = n.get('data') or {}
	return data.get('label') or n.get('label') or ''


def _n_params(n: Dict[str, Any]) -> Dict[str, Any]:
	data = n.get('data') or {}
	params = data.get('params') or n.get('params') or {}
	return params if isinstance(params, dict) else {}


def _indegree(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, int]:
	deg = {n['id']: 0 for n in nodes}
	for e in edges:
		if e.get('target') in deg:
			deg[e['target']] += 1
	return deg


def _adjacency(edges: List[Dict[str, Any]]) -> Dict[str, List[str]]:
	adj: Dict[str, List[str]] = {}
	for e in edges:
		src = e.get('source')
		tgt = e.get('target')
		if src is None or tgt is None:
			continue
		adj.setdefault(src, []).append(tgt)
	return adj


def topological_order(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Kahn's algorithm for topological ordering. If multiple valid orders exist, this picks a stable one."""
	node_by_id = {n['id']: n for n in nodes}
	deg = _indegree(nodes, edges)
	queue = [nid for nid, d in deg.items() if d == 0]
	ordered: List[Dict[str, Any]] = []
	adj = _adjacency(edges)
	while queue:
		# Stable order by label/type for determinism
		queue.sort(key=lambda nid: (_n_type(node_by_id[nid]), _n_label(node_by_id[nid])))
		nid = queue.pop(0)
		ordered.append(node_by_id[nid])
		for nb in adj.get(nid, []):
			deg[nb] -= 1
			if deg[nb] == 0:
				queue.append(nb)
	# Fallback: if cycle or missing edges, append any not included
	seen = {n['id'] for n in ordered}
	for n in nodes:
		if n['id'] not in seen:
			ordered.append(n)
	return ordered


# --------------------------------------
# CNN Code Generation (Keras Sequential)
# --------------------------------------

def generate_cnn_code(graph):
    """
    Generate Keras code for a CNN model from a graph definition.
    
    Args:
        graph: Dictionary containing 'nodes' and 'edges' defining the CNN architecture
        
    Returns:
        String containing the generated Python code
    """
    nodes = graph.get('nodes') or []
    edges = graph.get('edges') or []
    ordered = topological_order(nodes, edges)

    # Find dataset and input parameters
    dataset_params = {}
    input_params = {}
    for n in ordered:
        if _n_type(n) == 'dataset':
            dataset_params = _n_params(n)
        if _n_type(n) == 'inputImage':
            input_params = _n_params(n)

    in_w = int(input_params.get('width', 224))
    in_h = int(input_params.get('height', 224))
    in_c = int(input_params.get('channels', 3))
    batch_size = int(dataset_params.get('batch_size', 32))

    # Initialize code lines
    lines: List[str] = []
    lines.append('import tensorflow as tf')
    lines.append('from tensorflow.keras import layers, models, Sequential')
    lines.append('from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2, EfficientNetB0')
    lines.append('from tensorflow.keras.optimizers import Adam')
    lines.append('from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau')
    lines.append('from tensorflow.keras.preprocessing.image import ImageDataGenerator')
    lines.append('import numpy as np')
    lines.append('import matplotlib.pyplot as plt')
    lines.append('')
	
    data_augmentation_layers = []
    for n in ordered:
        if _n_type(n) == 'randomFlip':
            mode = _n_params(n).get('mode', 'horizontal')
            data_augmentation_layers.append(f"layers.RandomFlip('{mode}')")
        elif _n_type(n) == 'randomRotation':
            factor = _n_params(n).get('factor', 0.2)
            data_augmentation_layers.append(f"layers.RandomRotation({factor})")
        elif _n_type(n) == 'randomZoom':
            height_factor = _n_params(n).get('height_factor', 0.2)
            width_factor = _n_params(n).get('width_factor', 0.2)
            data_augmentation_layers.append(f"layers.RandomZoom(height_factor={height_factor}, width_factor={width_factor})")
        elif _n_type(n) == 'randomContrast':
            factor = _n_params(n).get('factor', 0.2)
            data_augmentation_layers.append(f"layers.RandomContrast({factor})")

    if data_augmentation_layers:
        lines.append('# Data Augmentation')
        lines.append('data_augmentation = Sequential([')
        for layer in data_augmentation_layers:
            lines.append(f'    {layer},')
        lines.append('    layers.Rescaling(1./255)')
        lines.append('])\n')
    else:
        # Basic preprocessing with just normalization
        lines.append('# Input preprocessing')
        lines.append('preprocessing = Sequential([')
        lines.append('    layers.Rescaling(1./255)')
        lines.append('])\n')

    # Check if this is a pre-trained model
    pretrained_type = None
    pretrained_trainable = False
    pretrained_params = {}
    
    for n in ordered:
        if _n_type(n) == 'pretrained':
            pretrained_type = _n_params(n).get('model', 'resnet50')
            pretrained_params = _n_params(n)
            pretrained_trainable = pretrained_params.get('trainable', False)
            break
    
    if pretrained_type:
        # Generate pre-trained model code
        lines.append(f'# Using pre-trained {pretrained_type}')
        
        # Different pre-trained models with options
        if pretrained_type == 'resnet50':
            lines.append(f'base_model = ResNet50(weights="imagenet", include_top=False, input_shape=({in_h}, {in_w}, {in_c}))')
        elif pretrained_type == 'vgg16':
            lines.append(f'base_model = VGG16(weights="imagenet", include_top=False, input_shape=({in_h}, {in_w}, {in_c}))')
        elif pretrained_type == 'mobilenet':
            lines.append(f'base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=({in_h}, {in_w}, {in_c}))')
        elif pretrained_type == 'efficientnet':
            lines.append(f'base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=({in_h}, {in_w}, {in_c}))')
        else:
            lines.append(f'base_model = ResNet50(weights="imagenet", include_top=False, input_shape=({in_h}, {in_w}, {in_c}))')
        
        lines.append('')
        lines.append('# Configure base model trainability')
        lines.append(f'base_model.trainable = {str(pretrained_trainable).lower()}')
        
        # Add fine-tuning options if the base model is trainable
        if pretrained_trainable:
            lines.append('')
            lines.append('# Fine-tuning: Unfreeze some top layers')
            lines.append('if hasattr(base_model, "layers"):')
            lines.append('    for layer in base_model.layers[-10:]:')
            lines.append('        if not isinstance(layer, layers.BatchNormalization):')
            lines.append('            layer.trainable = True')
        
        lines.append('')
        lines.append('# Create model with preprocessing and base model')
        lines.append('model = models.Sequential()')
        if use_data_augmentation:
            lines.append('model.add(data_augmentation)')
        else:
            lines.append('model.add(preprocessing)')
            
        lines.append('model.add(base_model)')
        lines.append('model.add(layers.GlobalAveragePooling2D())')
        
        # Add batch normalization if specified in input node
        input_node = next((n for n in ordered if _n_type(n) == 'inputImage'), None)
        if input_node and _n_params(input_node).get('batch_normalization', False):
            lines.append('model.add(layers.BatchNormalization())')
        
        # Add custom top layers
        for n in ordered:
            type_ = _n_type(n)
            params = _n_params(n)
            if type_ in ['dense', 'output']:
                if type_ == 'dense':
                    units = int(params.get('units') or 128)
                    activation = str(params.get('activation') or 'relu')
                    use_bias = params.get('use_bias', True)
                    kernel_initializer = params.get('kernel_initializer', 'glorot_uniform')
                    kernel_regularizer = params.get('kernel_regularizer')
                    bias_regularizer = params.get('bias_regularizer')
                    
                    # Build the layer string with optional parameters
                    layer_str = f"layers.Dense({units}, activation='{activation}'"
                    layer_str += f", use_bias={str(use_bias).lower()}"
                    layer_str += f", kernel_initializer='{kernel_initializer}'"
                    
                    if kernel_regularizer:
                        if kernel_regularizer == 'l1':
                            layer_str += ", kernel_regularizer=tf.keras.regularizers.l1(0.01)"
                        elif kernel_regularizer == 'l2':
                            layer_str += ", kernel_regularizer=tf.keras.regularizers.l2(0.01)"
                    
                    if bias_regularizer:
                        if bias_regularizer == 'l1':
                            layer_str += ", bias_regularizer=tf.keras.regularizers.l1(0.01)"
                        elif bias_regularizer == 'l2':
                            layer_str += ", bias_regularizer=tf.keras.regularizers.l2(0.01)"
                    
                    layer_str += ")"
                    lines.append(f'    {layer_str},')
                elif type_ == 'output':
                    classes = int(params.get('classes') or 10)
                    activation = str(params.get('activation') or 'softmax')
                    lines.append(f'    layers.Dense({classes}, activation="{activation}")')
        
        lines.append(')')
        lines.append('')
        
        # Set loss based on output
        for n in ordered:
            if _n_type(n) == 'output':
                classes = int(_n_params(n).get('classes') or 10)
                activation = str(_n_params(n).get('activation') or 'softmax')
                if activation == 'softmax' and classes > 1:
                    loss = 'sparse_categorical_crossentropy'
                elif activation == 'sigmoid' and classes == 1:
                    loss = 'binary_crossentropy'
                else:
                    loss = 'sparse_categorical_crossentropy'
                break
        else:
            loss = 'sparse_categorical_crossentropy'
        
        metrics = "['accuracy']"
        lines.append(f'model.compile(optimizer="adam", loss="{loss}", metrics={metrics})')
        lines.append('model.summary()')
        code = "\n".join(lines)
        return code

    # Standard sequential model generation
    lines.append('# Create model')
    lines.append('model = models.Sequential()')
    
    # Add input shape and preprocessing
    if use_data_augmentation:
        lines.append('model.add(data_augmentation)')
    else:
        lines.append(f'model.add(layers.Input(shape=({in_h}, {in_w}, {in_c})))')
        lines.append('model.add(layers.Rescaling(1./255))')
    
    # Initialize variables for model compilation
    loss = 'sparse_categorical_crossentropy'
    metrics = "['accuracy']"
    optimizer = 'Adam(learning_rate=0.001)'
    
    # Check if we need custom loss function
    for n in ordered:
        if _n_type(n) == 'output':
            classes = int(_n_params(n).get('classes') or 10)
            activation = str(_n_params(n).get('activation') or 'softmax')
            if classes == 1 and activation == 'sigmoid':
                loss = 'binary_crossentropy'
            elif classes > 2 and activation == 'softmax':
                loss = 'sparse_categorical_crossentropy'
            elif classes == 2 and activation == 'sigmoid':
                loss = 'binary_crossentropy'

        for n in ordered:
            type_ = _n_type(n)
            params = _n_params(n)
            if type_ == 'inputImage':
                continue
            if type_ == 'conv2d':
                filters = int(params.get('filters') or 32)
                kernel = _to_tuple_2(params.get('kernel') or '3x3', (3, 3))
                stride = _to_tuple_2(params.get('stride') or '1x1', (1, 1))
                padding = str(params.get('padding') or 'same')
                activation = str(params.get('activation') or 'relu')
                use_bias = params.get('use_bias', True)
                kernel_initializer = params.get('kernel_initializer', 'glorot_uniform')
                kernel_regularizer = params.get('kernel_regularizer')
                
                # Build the layer string with optional parameters
                layer_str = f"layers.Conv2D({filters}, {kernel}, strides={stride}, padding='{padding}'"
                layer_str += f", activation='{activation}'"
                layer_str += f", use_bias={str(use_bias).lower()}"
                layer_str += f", kernel_initializer='{kernel_initializer}'"
                
                if kernel_regularizer:
                    if kernel_regularizer == 'l1':
                        layer_str += ", kernel_regularizer=tf.keras.regularizers.l1(0.01)"
                    elif kernel_regularizer == 'l2':
                        layer_str += ", kernel_regularizer=tf.keras.regularizers.l2(0.01)"
                
                layer_str += ")"
                lines.append(f"    model.add({layer_str})")
            elif type_ == 'maxpool':
                pool = _to_tuple_2(params.get('pool') or '2x2', (2, 2))
                stride = _to_tuple_2(params.get('stride') or '2x2', pool)
                padding = str(params.get('padding') or 'valid')
                
                # Add MaxPooling2D with optional parameters
                lines.append(f"    model.add(layers.MaxPooling2D(pool_size={pool}, strides={stride}, padding='{padding}'))")
            elif type_ == 'batchnorm':
                momentum = float(params.get('momentum') or 0.99)
                epsilon = float(params.get('epsilon') or 0.001)
                center = params.get('center', True)
                scale = params.get('scale', True)
                
                lines.append('    model.add(layers.BatchNormalization(')
                lines.append(f'        momentum={momentum},')
                lines.append(f'        epsilon={epsilon},')
                lines.append(f'        center={str(center).lower()},')
                lines.append(f'        scale={str(scale).lower()}')
                lines.append('    ))')
            elif type_ == 'dropout':
                rate = float(params.get('rate') or 0.5)
                noise_shape = params.get('noise_shape')
                seed = params.get('seed')
                
                layer_str = f"layers.Dropout({rate}"
                if noise_shape:
                    layer_str += f", noise_shape={noise_shape}"
                if seed is not None:
                    layer_str += f", seed={seed}"
                layer_str += ")"
                
                lines.append(f"    model.add({layer_str})")
            elif type_ == 'flatten':
                lines.append('    model.add(layers.Flatten())')
            elif type_ == 'dense':
                units = int(params.get('units') or 128)
                activation = str(params.get('activation') or 'relu')
                use_bias = params.get('use_bias', True)
                kernel_initializer = params.get('kernel_initializer', 'glorot_uniform')
                kernel_regularizer = params.get('kernel_regularizer')
                bias_regularizer = params.get('bias_regularizer')
                
                # Build the layer string with optional parameters
                layer_str = f"layers.Dense({units}, activation='{activation}'"
                layer_str += f", use_bias={str(use_bias).lower()}"
                layer_str += f", kernel_initializer='{kernel_initializer}'"
                
                if kernel_regularizer:
                    if kernel_regularizer == 'l1':
                        layer_str += ", kernel_regularizer=tf.keras.regularizers.l1(0.01)"
                    elif kernel_regularizer == 'l2':
                        layer_str += ", kernel_regularizer=tf.keras.regularizers.l2(0.01)"
                
                if bias_regularizer:
                    if bias_regularizer == 'l1':
                        layer_str += ", bias_regularizer=tf.keras.regularizers.l1(0.01)"
                    elif bias_regularizer == 'l2':
                        layer_str += ", bias_regularizer=tf.keras.regularizers.l2(0.01)"
                
                layer_str += ")"
                lines.append(f"    model.add({layer_str})")
            elif type_ == 'output':
                classes = int(params.get('classes') or 10)
                activation = str(params.get('activation') or 'softmax')
                use_bias = params.get('use_bias', True)
                
                # Handle different output types
                if classes == 1 and activation == 'sigmoid':
                    loss = 'binary_crossentropy'
                    metrics = "['accuracy']"
                elif classes > 2 and activation == 'softmax':
                    loss = 'sparse_categorical_crossentropy'
                    metrics = "['sparse_categorical_accuracy']"
                elif classes == 2 and activation == 'sigmoid':
                    loss = 'binary_crossentropy'
                    metrics = "['accuracy']"
                else:
                    loss = 'sparse_categorical_crossentropy'
                    metrics = "['accuracy']"
                
                # Build output layer with options
                layer_str = f"layers.Dense({classes}, activation='{activation}'"
                layer_str += f", use_bias={str(use_bias).lower()}"
                layer_str += ", name='output')"
                
                lines.append(f"    model.add({layer_str})")
            # Handle output layer
            elif type_ == 'output':
                classes = int(params.get('classes') or 10)
                activation = str(params.get('activation') or 'softmax')
                use_bias = params.get('use_bias', True)
                
                # Handle different output types
                if classes == 1 and activation == 'sigmoid':
                    loss = 'binary_crossentropy'
                    metrics = "['accuracy']"
                elif classes > 2 and activation == 'softmax':
                    loss = 'sparse_categorical_crossentropy'
                    metrics = "['sparse_categorical_accuracy']"
                elif classes == 2 and activation == 'sigmoid':
                    loss = 'binary_crossentropy'
                    metrics = "['accuracy']"
                else:
                    loss = 'sparse_categorical_crossentropy'
                    metrics = "['accuracy']"
                
                # Build output layer with options
                layer_str = f"layers.Dense({classes}, activation='{activation}'"
                layer_str += f", use_bias={str(use_bias).lower()}"
                layer_str += ", name='output')"
                
                lines.append(f"    model.add({layer_str})")
    
    # Add model compilation
    lines.append('')
    lines.append('# Model compilation')
    lines.append(f'model.compile(')
    lines.append(f'    optimizer={optimizer},')
    lines.append(f'    loss="{loss}",')
    lines.append(f'    metrics={metrics}')
    lines.append(')')
    
    # Add model summary and visualization
    lines.append('')
    lines.append('# Model summary')
    lines.append('model.summary()')
    lines.append('')
    
    # Add code for model training preparation
    lines.append('''
# Prepare callbacks
callbacks = [
    ModelCheckpoint(
        filepath='best_model.keras',
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]''')

    # Add training code template
    lines.append('''
# Training parameters
epochs = 50
batch_size = ''' + str(batch_size) + '''

# Example training code (uncomment and modify as needed)

# --- Dataset Loading ---
dataset_name = dataset_params.get('name', 'CIFAR-10')
custom_path = dataset_params.get('custom_path', '')
val_split = dataset_params.get('validation_split', 20) / 100.0

lines.append(f'# Dataset: {dataset_name}')
if dataset_name == 'Custom':
    lines.append(f'# Make sure to set the correct path to your dataset')
    lines.append(f"DATASET_PATH = '{custom_path}'")
else:
    lines.append(f'# Using the {dataset_name} dataset')
    lines.append(f'# (Not implemented: code to automatically download and load {dataset_name})')
    lines.append(f"DATASET_PATH = 'path/to/{dataset_name.lower()}' # Please update this path")

lines.append(f"""
# The following code splits the data from DATASET_PATH into training and validation sets.
# It assumes that DATASET_PATH points to a directory with subdirectories for each class.
# You might need to adjust this logic if your dataset is structured differently,
# or if you have a separate directory for the test set.

# train_ds = tf.keras.utils.image_dataset_from_directory(
#     DATASET_PATH,
#     validation_split={val_split},
#     subset="training",
#     seed=123,
#     image_size=({in_h}, {in_w}),
#     batch_size={batch_size})

# val_ds = tf.keras.utils.image_dataset_from_directory(
#     DATASET_PATH,
#     validation_split={val_split},
#     subset="validation",
#     seed=123,
#     image_size=({in_h}, {in_w}),
#     batch_size={batch_size})
""")

    lines.append('''
# --- Model Training ---
# 
# # Train the model
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=epochs,
#     callbacks=callbacks,
#     verbose=1
# )

# Save the final model
# model.save('final_model.keras')
''')

    # Add code for evaluation and visualization
    lines.append('''
# Example evaluation and visualization
# def plot_training_history(history):
#     # Plot training & validation accuracy values
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('Model accuracy')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper left')

#     # Plot training & validation loss values
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper left')
#     plt.tight_layout()
#     plt.show()

# # Call the function with your training history
# # plot_training_history(history)
''')

    # Join all lines and return the code
    code = "\n".join(lines)
    return code


# -------------------------------------------
# RAG Code Generation (framework-agnostic)
# -------------------------------------------

def generate_rag_code(graph):
	nodes = graph.get('nodes') or []
	edges = graph.get('edges') or []
	ordered = topological_order(nodes, edges)

	# Collect parameters
	cfg: Dict[str, Dict[str, Any]] = {}
	for n in ordered:
		t = _n_type(n)
		p = _n_params(n)
		if t:
			cfg[t] = p

	source = (cfg.get('inputDocs') or {}).get('source', 'upload')
	path = (cfg.get('inputDocs') or {}).get('path', '')

	embed_model = (cfg.get('embed') or {}).get('model', 'text-embedding-3-small')
	embed_dim = int((cfg.get('embed') or {}).get('dim', 768))

	chunk_size = int((cfg.get('chunk') or {}).get('chunkSize', 800))
	overlap = int((cfg.get('chunk') or {}).get('overlap', 100))

	store = (cfg.get('vectorstore') or {}).get('store', 'faiss')
	index_name = (cfg.get('vectorstore') or {}).get('indexName', 'documents')

	top_k = int((cfg.get('retriever') or {}).get('topK', 5))
	reranker_model = (cfg.get('reranker') or {}).get('model', 'none')

	llm_model = (cfg.get('llm') or {}).get('model', 'gpt-4o-mini')
	temperature = float((cfg.get('llm') or {}).get('temperature', 0.7))

	lines: List[str] = []
	lines.append('from typing import List, Dict, Any')
	lines.append('')
	lines.append('def load_documents(source: str, path: str) -> List[str]:')
	lines.append("\t# TODO: Implement actual loaders (upload/URL/S3)")
	lines.append("\treturn ['Example document 1', 'Example document 2']")
	lines.append('')
	lines.append('def chunk_documents(docs: List[str], size: int, overlap: int) -> List[str]:')
	lines.append("\tchunks: List[str] = []")
	lines.append("\tfor d in docs:\n\t\tfor i in range(0, len(d), max(1, size - overlap)):\n\t\t\tchunks.append(d[i:i+size])")
	lines.append("\treturn chunks")
	lines.append('')
	lines.append('def embed_texts(texts: List[str], model_name: str, dim: int) -> List[List[float]]:')
	lines.append("\t# TODO: Call real embedding model: model_name")
	lines.append("\treturn [[0.0] * dim for _ in texts]")
	lines.append('')
	lines.append('def build_vectorstore(embeddings: List[List[float]], store: str, index_name: str) -> Any:')
	lines.append("\t# TODO: Build chosen vector store (faiss/chroma/milvus/pgvector)")
	lines.append("\treturn {'store': store, 'index_name': index_name, 'size': len(embeddings)}")
	lines.append('')
	lines.append('def retrieve(query: str, vs: Any, top_k: int) -> List[str]:')
	lines.append("\t# TODO: Similarity search in vector store")
	lines.append("\treturn [f'Chunk {i}' for i in range(top_k)]")
	lines.append('')
	lines.append('def rerank(chunks: List[str], model: str) -> List[str]:')
	lines.append("\t# TODO: Apply reranking model if any")
	lines.append("\treturn chunks if model == 'none' else chunks")
	lines.append('')
	lines.append('def answer(query: str, context: List[str], llm: str, temperature: float) -> str:')
	lines.append("\t# TODO: Call LLM with context")
	lines.append("\treturn f'Answer using {llm} with {len(context)} chunks'\n")
	lines.append('')
	lines.append('def main() -> None:')
	lines.append(f"\tDOC_SOURCE = '{source}'")
	lines.append(f"\tDOC_PATH = '{path}'")
	lines.append(f"\tEMBED_MODEL = '{embed_model}'")
	lines.append(f"\tEMBED_DIM = {embed_dim}")
	lines.append(f"\tCHUNK_SIZE = {chunk_size}")
	lines.append(f"\tOVERLAP = {overlap}")
	lines.append(f"\tSTORE = '{store}'")
	lines.append(f"\tINDEX_NAME = '{index_name}'")
	lines.append(f"\tTOP_K = {top_k}")
	lines.append(f"\tRERANKER = '{reranker_model}'")
	lines.append(f"\tLLM = '{llm_model}'")
	lines.append(f"\tTEMPERATURE = {temperature}")
	lines.append('')
	lines.append('\tdocs = load_documents(DOC_SOURCE, DOC_PATH)')
	lines.append('\tchunks = chunk_documents(docs, CHUNK_SIZE, OVERLAP)')
	lines.append('\tembeddings = embed_texts(chunks, EMBED_MODEL, EMBED_DIM)')
	lines.append('\tvs = build_vectorstore(embeddings, STORE, INDEX_NAME)')
	lines.append("\tresults = retrieve('example query', vs, TOP_K)")
	lines.append('\tresults = rerank(results, RERANKER)')
	lines.append("\tprint(answer('example query', results, LLM, TEMPERATURE))")
	lines.append('')
	lines.append("if __name__ == '__main__':")
	lines.append('\tmain()')
	code = "\n".join(lines)
	return code



@app.post('/validate')
def validate():
	data = request.get_json(force=True)
	graph = data.get('graph')
	builder_type = data.get('builder_type') or data.get('type')  # 'cnn' or 'rag'
	if not graph or builder_type not in ('cnn', 'rag'):
		return jsonify({"error": "invalid request"}), 400
	
	nodes = graph.get('nodes') or []
	edges = graph.get('edges') or []
	validation = validate_graph_structure(builder_type, nodes, edges)
	
	return jsonify(validation)



# -------------------------------
# Auth Endpoints (signup/login)
# -------------------------------

@app.post('/signup')
def signup():
	data = request.get_json(force=True)
	username = (data.get('username') or '').strip()
	email = (data.get('email') or '').strip().lower()
	password = data.get('password') or ''
	if not username or not email or not password:
		return jsonify({"error": "username, email and password are required"}), 400

	password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
	try:
		with get_db_connection() as conn:
			with conn.cursor(cursor_factory=RealDictCursor) as cur:
				cur.execute(
					"""
					INSERT INTO users (username, email, password_hash)
					VALUES (%s, %s, %s)
					RETURNING user_id, username, email, created_at
					""",
					(username, email, password_hash)
				)
				user = cur.fetchone()
				conn.commit()
		token = create_access_token(user)
		return jsonify({"user": user, "token": token, "message": "User created successfully"}), 201
	except errors.UniqueViolation:
		return jsonify({"error": "username or email already exists"}), 409
	except Exception as e:
		return jsonify({"error": str(e)}), 500


@app.post('/login')
def login():
	data = request.get_json(force=True)
	email = (data.get('email') or '').strip().lower()
	password = data.get('password') or ''
	if not email or not password:
		return jsonify({"error": "email and password are required"}), 400

	try:
		with get_db_connection() as conn:
			with conn.cursor(cursor_factory=RealDictCursor) as cur:
				cur.execute("SELECT user_id, username, email, password_hash FROM users WHERE email = %s", (email,))
				row = cur.fetchone()
				if not row:
					return jsonify({"error": "Invalid credentials"}), 401
				stored_hash = (row.get('password_hash') or '').encode('utf-8')
				if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
					user = {k: row[k] for k in ['user_id', 'username', 'email']}
					token = create_access_token(user)
					return jsonify({"message": "Login successful", "user": user, "token": token})
				else:
					return jsonify({"error": "Invalid credentials"}), 401
	except Exception as e:
		return jsonify({"error": str(e)}), 500


# -------------------------------
# Subscriptions
# -------------------------------

@app.get('/subscriptions')
def list_subscriptions():
	try:
		with get_db_connection() as conn:
			with conn.cursor(cursor_factory=RealDictCursor) as cur:
				cur.execute("SELECT subscription_id, name, price, features, created_at FROM subscriptions ORDER BY subscription_id")
				rows = cur.fetchall()
		return jsonify({"subscriptions": rows})
	except Exception as e:
		return jsonify({"error": str(e)}), 500


# -------------------------------
# User Subscription (current user)
# -------------------------------

@app.get('/subscription')
@auth_required
def get_my_subscription():
	try:
		with get_db_connection() as conn:
			with conn.cursor(cursor_factory=RealDictCursor) as cur:
				cur.execute(
					"""
					SELECT us.user_subscription_id, us.is_active, us.start_date, us.end_date,
					       s.subscription_id, s.name, s.price, s.features
					FROM user_subscriptions us
					JOIN subscriptions s ON s.subscription_id = us.subscription_id
					WHERE us.user_id = %s AND us.is_active = TRUE
					ORDER BY us.user_subscription_id DESC
					LIMIT 1
					""",
					(request.user['user_id'],)
				)
				row = cur.fetchone()
		return jsonify({"subscription": row})
	except Exception as e:
		return jsonify({"error": str(e)}), 500


@app.post('/subscribe')
@auth_required
def subscribe():
	data = request.get_json(force=True)
	plan = (data.get('plan') or '').strip().lower()
	if plan not in ('free', 'pro', 'enterprise'):
		return jsonify({"error": "invalid plan"}), 400
	try:
		with get_db_connection() as conn:
			with conn.cursor(cursor_factory=RealDictCursor) as cur:
				# find subscription id by name (case-insensitive)
				cur.execute("SELECT subscription_id FROM subscriptions WHERE lower(name) = %s", (plan,))
				sub = cur.fetchone()
				if not sub:
					# auto-create simple plan if not exists
					cur.execute(
						"INSERT INTO subscriptions(name, price, features) VALUES (%s, %s, %s) RETURNING subscription_id",
						(plan.capitalize(), 0 if plan=='free' else (19.99 if plan=='pro' else 99.0), None)
					)
					sub = cur.fetchone()
				# deactivate existing
				cur.execute("UPDATE user_subscriptions SET is_active = FALSE, end_date = CURRENT_TIMESTAMP WHERE user_id = %s AND is_active = TRUE", (request.user['user_id'],))
				# create new
				cur.execute(
					"""
					INSERT INTO user_subscriptions (user_id, subscription_id)
					VALUES (%s, %s)
					RETURNING user_subscription_id, user_id, subscription_id, start_date, end_date, is_active
					""",
					(request.user['user_id'], sub['subscription_id'])
				)
				row = cur.fetchone()
				conn.commit()
		return jsonify({"user_subscription": row}), 201
	except Exception as e:
		return jsonify({"error": str(e)}), 500


@app.post('/cancel-subscription')
@auth_required
def cancel_subscription():
	try:
		with get_db_connection() as conn:
			with conn.cursor(cursor_factory=RealDictCursor) as cur:
				cur.execute("UPDATE user_subscriptions SET is_active = FALSE, end_date = CURRENT_TIMESTAMP WHERE user_id = %s AND is_active = TRUE RETURNING user_subscription_id", (request.user['user_id'],))
				conn.commit()
		return jsonify({"cancelled": True})
	except Exception as e:
		return jsonify({"error": str(e)}), 500


@app.post('/subscriptions')
@auth_required
def create_subscription():
	data = request.get_json(force=True)
	name = (data.get('name') or '').strip()
	price = data.get('price', 0)
	features = data.get('features')
	if not name:
		return jsonify({"error": "name is required"}), 400
	try:
		with get_db_connection() as conn:
			with conn.cursor(cursor_factory=RealDictCursor) as cur:
				cur.execute(
					"""
					INSERT INTO subscriptions (name, price, features)
					VALUES (%s, %s, %s)
					RETURNING subscription_id, name, price, features, created_at
					""",
					(name, price, features)
				)
				row = cur.fetchone()
				conn.commit()
		return jsonify({"subscription": row}), 201
	except Exception as e:
		return jsonify({"error": str(e)}), 500


@app.get('/subscriptions/<int:subscription_id>')
def get_subscription(subscription_id: int):
	try:
		with get_db_connection() as conn:
			with conn.cursor(cursor_factory=RealDictCursor) as cur:
				cur.execute("SELECT subscription_id, name, price, features, created_at FROM subscriptions WHERE subscription_id = %s", (subscription_id,))
				row = cur.fetchone()
				if not row:
					return jsonify({"error": "not found"}), 404
		return jsonify({"subscription": row})
	except Exception as e:
		return jsonify({"error": str(e)}), 500


@app.put('/subscriptions/<int:subscription_id>')
@auth_required
def update_subscription(subscription_id: int):
	data = request.get_json(force=True)
	name = data.get('name')
	price = data.get('price')
	features = data.get('features')
	try:
		with get_db_connection() as conn:
			with conn.cursor(cursor_factory=RealDictCursor) as cur:
				cur.execute(
					"""
					UPDATE subscriptions
					SET name = COALESCE(%s, name),
					    price = COALESCE(%s, price),
					    features = COALESCE(%s, features)
					WHERE subscription_id = %s
					RETURNING subscription_id, name, price, features, created_at
					""",
					(name, price, features, subscription_id)
				)
				row = cur.fetchone()
				if not row:
					return jsonify({"error": "not found"}), 404
				conn.commit()
		return jsonify({"subscription": row})
	except Exception as e:
		return jsonify({"error": str(e)}), 500


@app.delete('/subscriptions/<int:subscription_id>')
@auth_required
def delete_subscription(subscription_id: int):
	try:
		with get_db_connection() as conn:
			with conn.cursor() as cur:
				cur.execute("DELETE FROM subscriptions WHERE subscription_id = %s", (subscription_id,))
				if cur.rowcount == 0:
					return jsonify({"error": "not found"}), 404
				conn.commit()
		return jsonify({"deleted": True})
	except Exception as e:
		return jsonify({"error": str(e)}), 500


@app.post('/users/<int:user_id>/subscriptions')
@auth_required
def assign_subscription(user_id: int):
	data = request.get_json(force=True)
	subscription_id = data.get('subscription_id')
	end_date = data.get('end_date')
	if not subscription_id:
		return jsonify({"error": "subscription_id is required"}), 400
	try:
		with get_db_connection() as conn:
			with conn.cursor(cursor_factory=RealDictCursor) as cur:
				cur.execute(
					"""
					INSERT INTO user_subscriptions (user_id, subscription_id, end_date)
					VALUES (%s, %s, %s)
					RETURNING user_subscription_id, user_id, subscription_id, start_date, end_date, is_active
					""",
					(user_id, subscription_id, end_date)
				)
				row = cur.fetchone()
				conn.commit()
		return jsonify({"user_subscription": row}), 201
	except Exception as e:
		return jsonify({"error": str(e)}), 500


@app.get('/users/<int:user_id>/subscriptions')
@auth_required
def list_user_subscriptions(user_id: int):
	try:
		with get_db_connection() as conn:
			with conn.cursor(cursor_factory=RealDictCursor) as cur:
				cur.execute(
					"""
					SELECT us.user_subscription_id, us.user_id, us.subscription_id, us.start_date, us.end_date, us.is_active,
					       s.name, s.price, s.features
					FROM user_subscriptions us
					JOIN subscriptions s ON s.subscription_id = us.subscription_id
					WHERE us.user_id = %s
					ORDER BY us.user_subscription_id DESC
					""",
					(user_id,)
				)
				rows = cur.fetchall()
		return jsonify({"subscriptions": rows})
	except Exception as e:
		return jsonify({"error": str(e)}), 500


# -------------------------------
# Chats and Messages
# -------------------------------

@app.post('/chats')
@auth_required
def create_chat():
	data = request.get_json(force=True)
	user_id = request.user['user_id']
	try:
		with get_db_connection() as conn:
			with conn.cursor(cursor_factory=RealDictCursor) as cur:
				cur.execute(
					"""
					INSERT INTO chats (user_id)
					VALUES (%s)
					RETURNING chat_id, user_id, created_at
					""",
					(user_id,)
				)
				row = cur.fetchone()
				conn.commit()
		return jsonify({"chat": row}), 201
	except Exception as e:
		return jsonify({"error": str(e)}), 500


@app.get('/users/<int:user_id>/chats')
@auth_required
def list_user_chats(user_id: int):
	if request.user['user_id'] != user_id:
		return jsonify({"error": "forbidden"}), 403
	try:
		with get_db_connection() as conn:
			with conn.cursor(cursor_factory=RealDictCursor) as cur:
				cur.execute("SELECT chat_id, user_id, created_at FROM chats WHERE user_id = %s ORDER BY chat_id DESC", (user_id,))
				rows = cur.fetchall()
		return jsonify({"chats": rows})
	except Exception as e:
		return jsonify({"error": str(e)}), 500


@app.post('/chats/<int:chat_id>/messages')
@auth_required
def add_message(chat_id: int):
	data = request.get_json(force=True)
	sender = (data.get('sender') or '').strip()
	message = data.get('message') or ''
	if sender not in ('user', 'ai'):
		return jsonify({"error": "sender must be 'user' or 'ai'"}), 400
	if not message:
		return jsonify({"error": "message is required"}), 400
	try:
		with get_db_connection() as conn:
			with conn.cursor(cursor_factory=RealDictCursor) as cur:
				# Verify chat ownership
				cur.execute("SELECT user_id FROM chats WHERE chat_id = %s", (chat_id,))
				ch = cur.fetchone()
				if not ch:
					return jsonify({"error": "chat not found"}), 404
				if ch['user_id'] != request.user['user_id']:
					return jsonify({"error": "forbidden"}), 403
				cur.execute(
					"""
					INSERT INTO messages (chat_id, sender, message)
					VALUES (%s, %s, %s)
					RETURNING message_id, chat_id, sender, message, created_at
					""",
					(chat_id, sender, message)
				)
				row = cur.fetchone()
				conn.commit()
		return jsonify({"message": row}), 201
	except Exception as e:
		return jsonify({"error": str(e)}), 500


@app.get('/chats/<int:chat_id>/messages')
@auth_required
def list_messages(chat_id: int):
	try:
		with get_db_connection() as conn:
			with conn.cursor(cursor_factory=RealDictCursor) as cur:
				# Verify chat ownership
				cur.execute("SELECT user_id FROM chats WHERE chat_id = %s", (chat_id,))
				ch = cur.fetchone()
				if not ch:
					return jsonify({"error": "chat not found"}), 404
				if ch['user_id'] != request.user['user_id']:
					return jsonify({"error": "forbidden"}), 403
				cur.execute(
					"SELECT message_id, chat_id, sender, message, created_at FROM messages WHERE chat_id = %s ORDER BY message_id",
					(chat_id,)
				)
				rows = cur.fetchall()
		return jsonify({"messages": rows})
	except Exception as e:
		return jsonify({"error": str(e)}), 500


# -------------------------------
# Simple Chat History (flat)
# -------------------------------

@app.get('/chats')
@auth_required
def list_chats_flat():
	try:
		with get_db_connection() as conn:
			with conn.cursor(cursor_factory=RealDictCursor) as cur:
				cur.execute(
					"SELECT id, role, message, created_at FROM chat_messages WHERE user_id = %s ORDER BY created_at ASC",
					(request.user['user_id'],)
				)
				rows = cur.fetchall()
		return jsonify({"chats": rows})
	except Exception as e:
		return jsonify({"error": str(e)}), 500


@app.post('/chats')
@auth_required
def add_chat_flat():
	data = request.get_json(force=True)
	role = (data.get('role') or '').strip()
	message = (data.get('message') or '').strip()
	if role not in ('user', 'assistant'):
		return jsonify({"error": "role must be 'user' or 'assistant'"}), 400
	if not message:
		return jsonify({"error": "message is required"}), 400
	try:
		with get_db_connection() as conn:
			with conn.cursor(cursor_factory=RealDictCursor) as cur:
				cur.execute(
					"""
					INSERT INTO chat_messages (user_id, role, message)
					VALUES (%s, %s, %s)
					RETURNING id, role, message, created_at
					""",
					(request.user['user_id'], role, message)
				)
				row = cur.fetchone()
				conn.commit()
		return jsonify({"message": row}), 201
	except Exception as e:
		return jsonify({"error": str(e)}), 500


@app.delete('/chats')
@auth_required
def clear_chats_flat():
	try:
		with get_db_connection() as conn:
			with conn.cursor() as cur:
				cur.execute("DELETE FROM chat_messages WHERE user_id = %s", (request.user['user_id'],))
				conn.commit()
		return jsonify({"cleared": True})
	except Exception as e:
		return jsonify({"error": str(e)}), 500


# -------------------------------
# Chat mode endpoints (placeholders)
# -------------------------------

@app.post('/chat-cnn')
@auth_required
def chat_cnn():
	# Placeholder for CNN-based chat processing
	return jsonify({"ok": True})


@app.post('/chat-rag')
@auth_required
def chat_rag():
	# Placeholder for RAG-based chat processing
	return jsonify({"ok": True})


# -------------------------------
# Users CRUD (self only)
# -------------------------------

@app.get('/users/me')
@auth_required
def get_me():
	return jsonify({"user": request.user})


@app.put('/users/me')
@auth_required
def update_me():
	data = request.get_json(force=True)
	username = (data.get('username') or '').strip() or None
	password = data.get('password') or None
	new_hash = None
	if password:
		new_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
	try:
		with get_db_connection() as conn:
			with conn.cursor(cursor_factory=RealDictCursor) as cur:
				cur.execute(
					"""
					UPDATE users
					SET username = COALESCE(%s, username),
					    password_hash = COALESCE(%s, password_hash)
					WHERE user_id = %s
					RETURNING user_id, username, email, created_at
					""",
					(username, new_hash, request.user['user_id'])
				)
				row = cur.fetchone()
				conn.commit()
		return jsonify({"user": row})
	except errors.UniqueViolation:
		return jsonify({"error": "username already exists"}), 409
	except Exception as e:
		return jsonify({"error": str(e)}), 500


# Backwards-compatibility alias
@app.post('/update-profile')
@auth_required
def update_profile_alias():
	return update_me()


@app.delete('/users/me')
@auth_required
def delete_me():
	try:
		with get_db_connection() as conn:
			with conn.cursor() as cur:
				cur.execute("DELETE FROM users WHERE user_id = %s", (request.user['user_id'],))
				conn.commit()
		return jsonify({"deleted": True})
	except Exception as e:
		return jsonify({"error": str(e)}), 500



def format_code_with_black(code_str: str) -> str:
    """Format Python code using black if available."""
    try:
        import black
        from black import FileMode
        return black.format_str(code_str, mode=FileMode(line_length=88))
    except ImportError:
        return code_str

@app.post('/generate')
def generate():
    try:
        print("\n=== New Code Generation Request ===")
        data = request.get_json(force=True)
        print(f"Received data keys: {list(data.keys())}")
        
        graph = data.get('graph')
        builder_type = data.get('builder_type') or data.get('type')  # 'cnn' or 'rag'
        print(f"Builder type: {builder_type}")
        
        # Input validation
        if not graph or builder_type not in ('cnn', 'rag'):
            return jsonify({
                "success": False,
                "error": "Invalid request: Missing or invalid graph/builder_type",
                "validation": {"errors": ["Missing or invalid graph/builder_type"], "warnings": []}
            }), 400
        
        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])
        
        # Validate graph structure
        validation = validate_graph_structure(builder_type, nodes, edges)
        
        # Return early if there are validation errors
        if validation.get('errors'):
            return jsonify({
                "success": False,
                "error": "Validation failed",
                "validation": validation,
                "code": ""
            }), 400
        
        # Generate code based on builder type
        try:
            # Add debug logging
            print(f"Starting code generation for builder_type: {builder_type}")
            print(f"Graph nodes: {len(graph.get('nodes', []))}, edges: {len(graph.get('edges', []))}")
            
            # Generate the code
            try:
                print(f"Starting code generation for type: {builder_type}")
                if builder_type == 'cnn':
                    print("Generating CNN code...")
                    code = generate_cnn_code(graph)
                else:
                    print("Generating RAG code...")
                    code = generate_rag_code(graph)
                
                print(f"Raw code generated. Length: {len(code) if code else 0} characters")
                
                if not code or not isinstance(code, str) or len(code.strip()) == 0:
                    print("Error: Empty or invalid code generated")
                    return jsonify({
                        "success": False,
                        "error": "Code generation resulted in empty output",
                        "validation": {"errors": ["Code generation failed: Empty output"], "warnings": []},
                        "code": ""
                    }), 400
                
                # Format the code if it exists
                try:
                    print("Formatting code...")
                    formatted_code = format_code_with_black(code)
                    if formatted_code and len(formatted_code) > 10:  # Basic validation
                        print("Code formatted successfully")
                        code = formatted_code
                    else:
                        print("Warning: Formatting returned empty or invalid code, using unformatted code")
                except Exception as format_err:
                    print(f"Code formatting warning: {str(format_err)}")
                    print("Using unformatted code")
                    # Continue with unformatted code if formatting fails
                
                print(f"Final code length: {len(code)} characters")
                
            except Exception as gen_error:
                error_msg = f"Error during code generation: {str(gen_error)}"
                print(error_msg)
                print(traceback.format_exc())
                return jsonify({
                    "success": False,
                    "error": error_msg,
                    "validation": {"errors": [f"Code generation failed: {error_msg}"], "warnings": []},
                    "code": ""
                }), 500
            
            return jsonify({
                "success": True,
                "code": code,
                "language": "python",
                "validation": validation
            })
            
        except Exception as gen_error:
            error_msg = str(gen_error)
            print(f"Code generation error: {error_msg}\n{traceback.format_exc()}")
            return jsonify({
                "success": False,
                "error": f"Code generation failed: {error_msg}",
                "validation": {"errors": [f"Generation error: {error_msg}"], "warnings": []},
                "code": ""
            }), 400
            
    except Exception as e:
        error_msg = str(e)
        print(f"Unexpected error in /generate: {error_msg}\n{traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {error_msg}",
            "validation": {"errors": ["An unexpected error occurred during code generation"], "warnings": []},
            "code": ""
        }), 500


@app.post('/run')
def run_code():
    try:
        data = request.get_json(force=True)
        code = data.get('code')
        
        if not code or not isinstance(code, str):
            return jsonify({
                "success": False,
                "error": "Missing or invalid code parameter"
            }), 400

        # Basic code validation
        if len(code.strip()) == 0:
            return jsonify({
                "success": False,
                "error": "Empty code provided"
            }), 400

        # Security: Check for potentially dangerous operations
        dangerous_patterns = [
            'import os', 'import sys', 'import subprocess',
            'os.system', 'subprocess.run', 'exec(', 'eval(',
            'open(', 'file(', 'import socket', 'import ctypes'
        ]
        
        if any(pattern in code for pattern in dangerous_patterns):
            return jsonify({
                "success": False,
                "error": "Code contains restricted operations"
            }), 403

        # Write code to a temp file and execute in a subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, 'user_code.py')
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(textwrap.dedent(code))
                
                # Execute with timeout
                proc = subprocess.run(
                    [sys.executable, file_path],
                    capture_output=True,
                    text=True,
                    timeout=30,  # Shorter timeout for security
                    cwd=tmpdir,  # Run in temp directory
                    env={"PYTHONPATH": "."}  # Limit Python path
                )
                
                return jsonify({
                    "success": True,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "timed_out": False
                })
                
            except subprocess.TimeoutExpired:
                return jsonify({
                    "success": False,
                    "error": "Execution timed out (30s limit)",
                    "timed_out": True
                }), 408
                
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"Execution failed: {str(e)}",
                    "timed_out": False
                }), 400
                
    except Exception as e:
        error_msg = str(e)
        print(f"Unexpected error in /run: {error_msg}\n{traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": "Internal server error during code execution"
        }), 500


if __name__ == '__main__':
    # Initialize DB schema on startup
    try:
        init_db_schema()
    except Exception as e:
        print(f"Warning: Could not initialize database schema: {e}")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=8000, debug=True)