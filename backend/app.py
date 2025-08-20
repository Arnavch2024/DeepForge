from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import sys
import tempfile
import textwrap
import json
import os

app = Flask(__name__)
CORS(app)

# Simple health check
@app.get('/health')
def health():
	return jsonify({"status": "ok"})


def generate_cnn_code(graph):
	# Placeholder: translate nodes/edges to Keras/PyTorch code. Return as string.
	# Expect graph = { "nodes": [...], "edges": [...] }
	code = """
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential()
# TODO: dynamically generated from graph
model.add(layers.Input(shape=(224, 224, 3)))
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
""".strip()
	return code


def generate_rag_code(graph):
	# Placeholder: translate nodes/edges to RAG pipeline code
	code = """
# Example RAG skeleton
from typing import List

def embed_texts(texts: List[str]) -> List[List[float]]:
	# TODO: call embedding model
	return [[0.0] * 768 for _ in texts]

print('RAG pipeline placeholder')
""".strip()
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