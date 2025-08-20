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

export default function RAGBuilder() {
  return (
    <BaseBuilder
      title="RAG Builder"
      palette={ragPalette}
      storageKey="deepforge:builder:rag:v1"
      schemas={ragSchemas}
      builderType="rag"
    />
  );
} 