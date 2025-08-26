import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/builder.css';
import BackButton from '../components/BackButton';

export default function BuilderHub() {
  return (
    <div className="builder-hub">
      <BackButton />
      <h1 className="hub-title">Choose a Builder</h1>
      <div className="hub-cards">
        <Link className="hub-card" to="/builder/cnn">
          <div className="hub-card-title">CNN Builder</div>
          <div className="hub-card-desc">Design convolutional networks visually</div>
        </Link>
        <Link className="hub-card" to="/builder/rag">
          <div className="hub-card-title">RAG Builder</div>
          <div className="hub-card-desc">Compose retrieval-augmented generation pipelines</div>
        </Link>
      </div>
    </div>
  );
} 