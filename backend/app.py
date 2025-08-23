from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import sys
import tempfile
import textwrap
import json
import os
from typing import Dict, List, Tuple, Any, Optional

app = Flask(__name__)
CORS(app)

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

def validate_cnn_graph(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[str]:
	"""Validate CNN graph structure and return list of errors/warnings"""
	errors = []
	warnings = []
	
	if not nodes:
		errors.append("No nodes found in the graph")
		return errors
	
	# Check for required nodes
	node_types = [_n_type(n) for n in nodes]
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
	nodes = graph.get('nodes') or []
	edges = graph.get('edges') or []
	ordered = topological_order(nodes, edges)

	# Defaults
	in_w, in_h, in_c = 224, 224, 3
	for n in ordered:
		t = _n_type(n)
		p = _n_params(n)
		if t == 'inputImage':
			in_w = int(p.get('width') or in_w)
			in_h = int(p.get('height') or in_h)
			in_c = int(p.get('channels') or in_c)
			break

	lines: List[str] = []
	lines.append('import tensorflow as tf')
	lines.append('from tensorflow.keras import layers, models')
	lines.append('from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2')
	lines.append('')
	
	# Check if this is a pre-trained model
	pretrained_type = None
	for n in ordered:
		if _n_type(n) == 'pretrained':
			pretrained_type = _n_params(n).get('model', 'resnet50')
			break
	
	if pretrained_type:
		# Generate pre-trained model code
		lines.append(f'# Using pre-trained {pretrained_type}')
		if pretrained_type == 'resnet50':
			lines.append(f'base_model = ResNet50(weights="imagenet", include_top=False, input_shape=({in_h}, {in_w}, {in_c}))')
		elif pretrained_type == 'vgg16':
			lines.append(f'base_model = VGG16(weights="imagenet", include_top=False, input_shape=({in_h}, {in_w}, {in_c}))')
		elif pretrained_type == 'mobilenet':
			lines.append(f'base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=({in_h}, {in_w}, {in_c}))')
		else:
			lines.append(f'base_model = ResNet50(weights="imagenet", include_top=False, input_shape=({in_h}, {in_w}, {in_c}))')
		
		lines.append('')
		lines.append('# Freeze base model layers')
		lines.append('base_model.trainable = False')
		lines.append('')
		lines.append('model = models.Sequential([')
		lines.append('    base_model,')
		lines.append('    layers.GlobalAveragePooling2D(),')
		
		# Add custom top layers
		for n in ordered:
			type_ = _n_type(n)
			params = _n_params(n)
			if type_ in ['dense', 'output']:
				if type_ == 'dense':
					units = int(params.get('units') or 128)
					activation = str(params.get('activation') or 'relu')
					lines.append(f"    layers.Dense({units}, activation='{activation}'),")
				elif type_ == 'output':
					classes = int(params.get('classes') or 10)
					activation = str(params.get('activation') or 'softmax')
					lines.append(f"    layers.Dense({classes}, activation='{activation}')")
		
		lines.append('])')
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
		lines.append(f"model.compile(optimizer='adam', loss='{loss}', metrics={metrics})")
		lines.append('model.summary()')
		code = "\n".join(lines)
		return code

	# Standard sequential model generation
	lines.append('model = models.Sequential()')
	lines.append(f'model.add(layers.Input(shape=({in_h}, {in_w}, {in_c}))')

	loss = 'sparse_categorical_crossentropy'
	metrics = "['accuracy']"

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
			lines.append(f"model.add(layers.Conv2D({filters}, {kernel}, strides={stride}, padding='{padding}', activation='{activation}'))")
		elif type_ == 'maxpool':
			pool = _to_tuple_2(params.get('pool') or '2x2', (2, 2))
			stride = _to_tuple_2(params.get('stride') or '2x2', pool)
			lines.append(f'model.add(layers.MaxPooling2D(pool_size={pool}, strides={stride}))')
		elif type_ == 'batchnorm':
			momentum = float(params.get('momentum') or 0.99)
			epsilon = float(params.get('epsilon') or 0.001)
			lines.append(f'model.add(layers.BatchNormalization(momentum={momentum}, epsilon={epsilon}))')
		elif type_ == 'dropout':
			rate = float(params.get('rate') or 0.5)
			lines.append(f'model.add(layers.Dropout({rate}))')
		elif type_ == 'flatten':
			lines.append('model.add(layers.Flatten())')
		elif type_ == 'dense':
			units = int(params.get('units') or 128)
			activation = str(params.get('activation') or 'relu')
			lines.append(f"model.add(layers.Dense({units}, activation='{activation}'))")
		elif type_ == 'output':
			classes = int(params.get('classes') or 10)
			activation = str(params.get('activation') or 'softmax')
			lines.append(f"model.add(layers.Dense({classes}, activation='{activation}'))")
			if activation == 'softmax' and classes > 1:
				loss = 'sparse_categorical_crossentropy'
			elif activation == 'sigmoid' and classes == 1:
				loss = 'binary_crossentropy'

	lines.append('')
	lines.append(f"model.compile(optimizer='adam', loss='{loss}', metrics={metrics})")
	lines.append('model.summary()')
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
	builder_type = data.get('type')  # 'cnn' or 'rag'
	if not graph or builder_type not in ('cnn', 'rag'):
		return jsonify({"error": "invalid request"}), 400
	
	nodes = graph.get('nodes') or []
	edges = graph.get('edges') or []
	validation = validate_graph_structure(builder_type, nodes, edges)
	
	return jsonify(validation)


@app.post('/generate')
def generate():
	data = request.get_json(force=True)
	graph = data.get('graph')
	builder_type = data.get('type')  # 'cnn' or 'rag'
	if not graph or builder_type not in ('cnn', 'rag'):
		return jsonify({"error": "invalid request"}), 400
	
	# Validate graph structure
	nodes = graph.get('nodes') or []
	edges = graph.get('edges') or []
	validation = validate_graph_structure(builder_type, nodes, edges)
	
	# Generate code
	if builder_type == 'cnn':
		code = generate_cnn_code(graph)
	else:
		code = generate_rag_code(graph)
	
	return jsonify({
		"code": code,
		"validation": validation
	})


@app.post('/run')
def run_code():
	data = request.get_json(force=True)
	code = data.get('code')
	if not code:
		return jsonify({"error": "missing code"}), 400

	# Write code to a temp file and execute in a subprocess (python)
	with tempfile.TemporaryDirectory() as tmpdir:
		file_path = os.path.join(tmpdir, 'user_code.py')
		with open(file_path, 'w', encoding='utf-8') as f:
			f.write(textwrap.dedent(code))
		try:
			proc = subprocess.run([sys.executable, file_path], capture_output=True, text=True, timeout=120)
			return jsonify({
				"returncode": proc.returncode,
				"stdout": proc.stdout,
				"stderr": proc.stderr,
			})
		except subprocess.TimeoutExpired:
			return jsonify({"error": "execution timeout"}), 408
		except Exception as e:
			return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8000, debug=True) 