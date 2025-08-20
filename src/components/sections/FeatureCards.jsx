import React from 'react';
import './FeatureCards.css';

const FeatureCards = () => {
  const cnnFeatures = [
    'Pre-built CNN architectures and templates',
    'Real-time model optimization suggestions',
    'Interactive layer visualization and debugging',
    'Performance monitoring and improvement tips',
    'Dataset preparation and augmentation guidance'
  ];

  const ragFeatures = [
    'Smart document indexing and chunking strategies',
    'Vector database integration and optimization',
    'Prompt engineering best practices',
    'Retrieval quality assessment and tuning',
    'End-to-end pipeline templates and examples'
  ];

  return (
    <section className="features-section" id="features">
      <div className="features-container">
        <h2 className="section-title">Choose Your AI Journey</h2>
        <p className="section-subtitle">
          Whether you're building intelligent document systems or computer vision models, 
          we've got the tools and guidance to accelerate your development.
        </p>
        
        <div className="features-grid">
          {/* CNN Card */}
          <div className="cnn-card slide-up">
            <div className="card-icon-cnn">🧠</div>
            <h3 className="card-title">Convolutional Neural Networks</h3>
            <p className="card-description">
              Build powerful computer vision models with guided assistance. From image 
              classification to object detection, get real-time code suggestions and 
              architectural recommendations.
            </p>
            <ul className="card-features">
              {cnnFeatures.map((feature, index) => (
                <li key={index} className="card-feature">{feature}</li>
              ))}
            </ul>
            <button 
              className="card-cta cnn-cta" 
              onClick={() => alert('CNN Builder clicked!')}
            >
              Start Building CNNs
            </button>
          </div>

          {/* RAG Card */}
          <div className="rag-card slide-up">
            <div className="card-icon-rag">📚</div>
            <h3 className="card-title">Retrieval-Augmented Generation</h3>
            <p className="card-description">
              Create intelligent knowledge systems that combine retrieval with generation. 
              Build chatbots, Q&A systems, and document analyzers with AI-powered assistance.
            </p>
            <ul className="card-features">
              {ragFeatures.map((feature, index) => (
                <li key={index} className="card-feature">{feature}</li>
              ))}
            </ul>
            <button 
              className="card-cta rag-cta" 
              onClick={() => alert('RAG Builder clicked!')}
            >
              Start Building RAG
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default FeatureCards;