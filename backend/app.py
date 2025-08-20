from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import sys
import tempfile
import textwrap
import json
import os
from typing import Dict, List, Tuple, Any

app = Flask(__name__)
CORS(app)

# Simple health check
@app.get('/health')
def health():
	return jsonify({"status": "ok"})


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
		queue.sort(key=lambda nid: (node_by_id[nid].get('data', {}).get('type', ''), node_by_id[nid].get('data', {}).get('label', '')))
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
		t = (n.get('data') or {}).get('type')
		p = (n.get('data') or {}).get('params') or {}
		if t == 'inputImage':
			in_w = int(p.get('width') or in_w)
			in_h = int(p.get('height') or in_h)
			in_c = int(p.get('channels') or in_c)
			break

	lines: List[str] = []
	lines.append('import tensorflow as tf')
	lines.append('from tensorflow.keras import layers, models')
	lines.append('')
	lines.append('model = models.Sequential()')
	lines.append(f'model.add(layers.Input(shape=({in_h}, {in_w}, {in_c})))')

	loss = 'sparse_categorical_crossentropy'
	metrics = "['accuracy']"

	for n in ordered:
		type_ = (n.get('data') or {}).get('type')
		params = (n.get('data') or {}).get('params') or {}
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
		t = (n.get('data') or {}).get('type')
		p = (n.get('data') or {}).get('params') or {}
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


@app.post('/generate')
def generate():
	data = request.get_json(force=True)
	graph = data.get('graph')
	builder_type = data.get('type')  # 'cnn' or 'rag'
	if not graph or builder_type not in ('cnn', 'rag'):
		return jsonify({"error": "invalid request"}), 400
	if builder_type == 'cnn':
		code = generate_cnn_code(graph)
	else:
		code = generate_rag_code(graph)
	return jsonify({"code": code})


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