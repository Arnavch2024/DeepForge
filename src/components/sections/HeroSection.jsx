import React from 'react';
import './HeroSection.css';

const HeroSection = () => {
  return (
    <section className="hero-section" id="home">
      <div className="hero-container fade-in">
        <h1 className="hero-title">
          Build RAG & CNN Projects Without Barriers
        </h1>
        
        <p className="hero-subtitle">
          Your AI-powered coding copilot that provides real-time suggestions, 
          optimizations, and explanations. Bridge the gap between AI theory 
          and hands-on implementation with intelligent assistance.
        </p>
        
        <div className="hero-cta">
          <button 
            className="btn-primary" 
            onClick={() => alert('Get Started clicked!')}
          >
            Get Started Free
          </button>
          <button 
            className="btn-secondary" 
            onClick={() => alert('Demo clicked!')}
          >
            Watch Demo
          </button>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;