import React from 'react';
import BaseBuilder from '../components/builder/BaseBuilder.jsx';

const ragPalette = [
  { type: 'inputDocs', label: 'Input Docs' },
  { type: 'embed', label: 'Embedder' },
  { type: 'chunk', label: 'Chunker' },
  { type: 'vectorstore', label: 'Vector Store' },
  { type: 'retriever', label: 'Retriever' },
  { type: 'reranker', label: 'Reranker' },
  { type: 'llm', label: 'LLM' },
  { type: 'output', label: 'Output' },
];

const ragSchemas = {
  inputDocs: {
    title: 'Document Input',
    description: 'Choose how documents get into your pipeline. You can upload files, point to web URLs, or reference cloud storage like S3. The builder will use these docs for indexing and answering questions.',
    fields: [
      { key: 'source', label: 'Source', type: 'select', options: ['upload', 'url', 's3'], default: 'upload', help: 'Upload files directly, fetch from a URL, or read from S3.' },
      { key: 'path', label: 'Path/URL', type: 'text', default: '', help: 'Provide a local path, a full http(s) URL, or an S3 URI like s3://bucket/path' },
    ]
  },
  embed: {
    title: 'Embedder',
    description: 'Transforms text into vectors so similar text becomes close in vector space. Higher-dimensional embeddings can capture more nuance but are heavier to store and search.',
    fields: [
      { key: 'model', label: 'Model', type: 'select', options: ['text-embedding-3-small', 'text-embedding-3-large', 'bge-small'], default: 'text-embedding-3-small', help: 'Pick based on cost/quality. Large models deliver higher quality but are slower/costlier.' },
      { key: 'dim', label: 'Dimensions', type: 'number', default: 768, help: 'Vector size. Must match the embedder; don’t change unless you know the exact dimension.' },
    ]
  },
  chunk: {
    title: 'Chunker',
    description: 'Splits documents into smaller overlapping chunks so the retriever can find the most relevant pieces. Overlap helps preserve context across chunk boundaries.',
    fields: [
      { key: 'chunkSize', label: 'Chunk Size', type: 'number', default: 800, help: 'Larger chunks include more context but can add noise; smaller chunks are more precise. Typical: 500–1000.' },
      { key: 'overlap', label: 'Overlap', type: 'number', default: 100, help: 'How much adjacent chunks share. 50–200 is common to avoid losing context.' },
    ]
  },
  vectorstore: {
    title: 'Vector Store',
    description: 'Stores embeddings for fast similarity search. Choose a backend that fits your scale and deployment (in-memory local store vs. server DB).',
    fields: [
      { key: 'store', label: 'Store', type: 'select', options: ['faiss', 'chroma', 'milvus', 'pgvector'], default: 'faiss', help: 'FAISS/Chroma for local prototyping, Milvus/PGVector for production-scale.' },
      { key: 'indexName', label: 'Index Name', type: 'text', default: 'documents', help: 'Logical collection name used to group your embeddings.' },
    ]
  },
  retriever: {
    title: 'Retriever',
    description: 'Finds the K most relevant chunks for a user query using vector similarity. Balance K to trade off completeness and speed.',
    fields: [
      { key: 'topK', label: 'Top K', type: 'number', default: 5, help: 'How many chunks to return. 3–10 is typical; increase if answers miss context.' },
    ]
  },
  reranker: {
    title: 'Reranker',
    description: 'Optionally reorders retrieved chunks using a specialized model to improve answer quality. Useful when the retriever returns many candidates.',
    fields: [
      { key: 'model', label: 'Model', type: 'select', options: ['none', 'bge-reranker', 'colbert'], default: 'none', help: 'If unsure, start with none. Add a reranker when recall is high but precision is low.' },
    ]
  },
  llm: {
    title: 'LLM',
    description: 'Generates the final response conditioned on the retrieved context. Temperature controls creativity vs. precision. Lower values produce more factual answers.',
    fields: [
      { key: 'model', label: 'Model', type: 'select', options: ['gpt-4o-mini', 'gpt-4.1', 'llama3.1'], default: 'gpt-4o-mini', help: 'Pick a model that fits your latency/cost requirements.' },
      { key: 'temperature', label: 'Temperature', type: 'number', default: 0.7, help: '0.0–0.3 = factual; 0.4–0.7 = balanced; 0.8+ = creative but riskier.' },
    ]
  },
  output: {
    title: 'Output',
    description: 'Represents the final answer produced by the LLM. You can post-process or format this as needed for your app UI.',
    fields: []
  },
};

// RAG Presets
const ragPresets = [
  {
    id: 'basic-rag',
    name: 'Basic RAG',
    build: () => {
      const nodes = [
        { id: '1', type: 'default', position: { x: 100, y: 100 }, data: { label: 'Input Docs', type: 'inputDocs', params: { source: 'upload', format: 'pdf' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '2', type: 'default', position: { x: 100, y: 200 }, data: { label: 'Chunker', type: 'chunk', params: { size: 500, overlap: 50, strategy: 'fixed' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '3', type: 'default', position: { x: 100, y: 300 }, data: { label: 'Embedder', type: 'embed', params: { model: 'sentence-transformers/all-MiniLM-L6-v2', dimension: 384 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '4', type: 'default', position: { x: 100, y: 400 }, data: { label: 'Vector Store', type: 'vectorstore', params: { type: 'chroma', index: 'cosine' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '5', type: 'default', position: { x: 100, y: 500 }, data: { label: 'Retriever', type: 'retriever', params: { topK: 5, threshold: 0.7 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '6', type: 'default', position: { x: 100, y: 600 }, data: { label: 'LLM', type: 'llm', params: { model: 'gpt-3.5-turbo', temperature: 0.1, maxTokens: 512 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '7', type: 'default', position: { x: 100, y: 700 }, data: { label: 'Output', type: 'output', params: { format: 'json', streaming: 'false' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } }
      ];
      const edges = [
        { id: 'e1-2', source: '1', target: '2', animated: true },
        { id: 'e2-3', source: '2', target: '3', animated: true },
        { id: 'e3-4', source: '3', target: '4', animated: true },
        { id: 'e4-5', source: '4', target: '5', animated: true },
        { id: 'e5-6', source: '5', target: '6', animated: true },
        { id: 'e6-7', source: '6', target: '7', animated: true }
      ];
      return { nodes, edges };
    }
  },
  {
    id: 'rag-reranker',
    name: 'RAG + Reranker',
    build: () => {
      const nodes = [
        { id: '1', type: 'default', position: { x: 100, y: 100 }, data: { label: 'Input Docs', type: 'inputDocs', params: { source: 'upload', format: 'pdf' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '2', type: 'default', position: { x: 100, y: 200 }, data: { label: 'Chunker', type: 'chunk', params: { size: 300, overlap: 50, strategy: 'semantic' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '3', type: 'default', position: { x: 100, y: 300 }, data: { label: 'Embedder', type: 'embed', params: { model: 'text-embedding-ada-002', dimension: 1536 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '4', type: 'default', position: { x: 100, y: 400 }, data: { label: 'Vector Store', type: 'vectorstore', params: { type: 'pinecone', index: 'cosine' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '5', type: 'default', position: { x: 100, y: 500 }, data: { label: 'Retriever', type: 'retriever', params: { topK: 10, threshold: 0.6 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '6', type: 'default', position: { x: 100, y: 600 }, data: { label: 'Reranker', type: 'reranker', params: { model: 'cross-encoder/ms-marco-MiniLM-L-12-v2', topK: 5 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '7', type: 'default', position: { x: 100, y: 700 }, data: { label: 'LLM', type: 'llm', params: { model: 'gpt-4', temperature: 0.0, maxTokens: 1024 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '8', type: 'default', position: { x: 100, y: 800 }, data: { label: 'Output', type: 'output', params: { format: 'json', streaming: 'true' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } }
      ];
      const edges = [
        { id: 'e1-2', source: '1', target: '2', animated: true },
        { id: 'e2-3', source: '2', target: '3', animated: true },
        { id: 'e3-4', source: '3', target: '4', animated: true },
        { id: 'e4-5', source: '4', target: '5', animated: true },
        { id: 'e5-6', source: '5', target: '6', animated: true },
        { id: 'e6-7', source: '6', target: '7', animated: true },
        { id: 'e7-8', source: '7', target: '8', animated: true }
      ];
      return { nodes, edges };
    }
  },
  {
    id: 'enterprise-rag',
    name: 'Enterprise RAG',
    build: () => {
      const nodes = [
        { id: '1', type: 'default', position: { x: 100, y: 100 }, data: { label: 'Input Docs', type: 'inputDocs', params: { source: 's3', format: 'pdf' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '2', type: 'default', position: { x: 100, y: 200 }, data: { label: 'Chunker', type: 'chunk', params: { size: 200, overlap: 20, strategy: 'semantic' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '3', type: 'default', position: { x: 100, y: 300 }, data: { label: 'Embedder', type: 'embed', params: { model: 'text-embedding-3-large', dimension: 3072 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '4', type: 'default', position: { x: 100, y: 400 }, data: { label: 'Vector Store', type: 'vectorstore', params: { type: 'weaviate', index: 'cosine' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '5', type: 'default', position: { x: 100, y: 500 }, data: { label: 'Retriever', type: 'retriever', params: { topK: 20, threshold: 0.5 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '6', type: 'default', position: { x: 100, y: 600 }, data: { label: 'Reranker', type: 'reranker', params: { model: 'cross-encoder/ms-marco-MiniLM-L-12-v2', topK: 8 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '7', type: 'default', position: { x: 100, y: 700 }, data: { label: 'LLM', type: 'llm', params: { model: 'gpt-4o', temperature: 0.0, maxTokens: 2048 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '8', type: 'default', position: { x: 100, y: 800 }, data: { label: 'Output', type: 'output', params: { format: 'json', streaming: 'true' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } }
      ];
      const edges = [
        { id: 'e1-2', source: '1', target: '2', animated: true },
        { id: 'e2-3', source: '2', target: '3', animated: true },
        { id: 'e3-4', source: '3', target: '4', animated: true },
        { id: 'e4-5', source: '4', target: '5', animated: true },
        { id: 'e5-6', source: '5', target: '6', animated: true },
        { id: 'e6-7', source: '6', target: '7', animated: true },
        { id: 'e7-8', source: '7', target: '8', animated: true }
      ];
      return { nodes, edges };
    }
  },
  {
    id: 'multimodal-rag',
    name: 'Multimodal RAG',
    build: () => {
      const nodes = [
        { id: '1', type: 'default', position: { x: 100, y: 100 }, data: { label: 'Input Docs', type: 'inputDocs', params: { source: 'upload', format: 'mixed' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '2', type: 'default', position: { x: 100, y: 200 }, data: { label: 'Chunker', type: 'chunk', params: { size: 400, overlap: 40, strategy: 'hybrid' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '3', type: 'default', position: { x: 100, y: 300 }, data: { label: 'Embedder', type: 'embed', params: { model: 'text-embedding-3-large', dimension: 3072 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '4', type: 'default', position: { x: 100, y: 400 }, data: { label: 'Vector Store', type: 'vectorstore', params: { type: 'qdrant', index: 'cosine' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '5', type: 'default', position: { x: 100, y: 500 }, data: { label: 'Retriever', type: 'retriever', params: { topK: 15, threshold: 0.6 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '6', type: 'default', position: { x: 100, y: 600 }, data: { label: 'Reranker', type: 'reranker', params: { model: 'cross-encoder/ms-marco-MiniLM-L-12-v2', topK: 6 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '7', type: 'default', position: { x: 100, y: 700 }, data: { label: 'LLM', type: 'llm', params: { model: 'gpt-4o', temperature: 0.1, maxTokens: 1536 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '8', type: 'default', position: { x: 100, y: 800 }, data: { label: 'Output', type: 'output', params: { format: 'markdown', streaming: 'true' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } }
      ];
      const edges = [
        { id: 'e1-2', source: '1', target: '2', animated: true },
        { id: 'e2-3', source: '2', target: '3', animated: true },
        { id: 'e3-4', source: '3', target: '4', animated: true },
        { id: 'e4-5', source: '4', target: '5', animated: true },
        { id: 'e5-6', source: '5', target: '6', animated: true },
        { id: 'e6-7', source: '6', target: '7', animated: true },
        { id: 'e7-8', source: '7', target: '8', animated: true }
      ];
      return { nodes, edges };
    }
  }
];

export default function RAGBuilder() {
  return (
    <BaseBuilder
      title="RAG Builder"
      palette={ragPalette}
      storageKey="deepforge:builder:rag:v1"
      schemas={ragSchemas}
      builderType="rag"
      presets={ragPresets}
    />
  );
} 