import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import './HeroSection.css';

const HeroSection = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('cnn');
  const demoSectionRef = useRef(null);
  const cursorGlowRef = useRef(null);

  const handleGetStarted = () => {
    navigate('/builder');
  };

  const handleWatchDemo = () => {
    // Navigate to the selected builder based on active tab
    navigate(`/builder/${activeTab}`);
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
  };

  // Cursor tracking effect
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (demoSectionRef.current && cursorGlowRef.current) {
        const rect = demoSectionRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        cursorGlowRef.current.style.left = `${x}px`;
        cursorGlowRef.current.style.top = `${y}px`;
      }
    };

    const handleMouseEnter = () => {
      if (cursorGlowRef.current) {
        cursorGlowRef.current.style.opacity = '1';
      }
    };

    const handleMouseLeave = () => {
      if (cursorGlowRef.current) {
        cursorGlowRef.current.style.opacity = '0';
      }
    };

    const demoSection = demoSectionRef.current;
    if (demoSection) {
      demoSection.addEventListener('mousemove', handleMouseMove);
      demoSection.addEventListener('mouseenter', handleMouseEnter);
      demoSection.addEventListener('mouseleave', handleMouseLeave);
    }

    return () => {
      if (demoSection) {
        demoSection.removeEventListener('mousemove', handleMouseMove);
        demoSection.removeEventListener('mouseenter', handleMouseEnter);
        demoSection.removeEventListener('mouseleave', handleMouseLeave);
      }
    };
  }, []);

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
            onClick={handleGetStarted}
          >
            Get Started Free
          </button>
          
          <div className="demo-section" ref={demoSectionRef}>
            {/* Interactive Cursor Glow */}
            <div className="cursor-glow" ref={cursorGlowRef}></div>
            
            {/* Animated Background Elements */}
            <div className="animated-border"></div>
            <div className="floating-particle"></div>
            <div className="floating-particle"></div>
            <div className="floating-particle"></div>
            <div className="floating-particle"></div>
            
            {/* Floating Orbs */}
            <div className="floating-orb"></div>
            <div className="floating-orb"></div>
            <div className="floating-orb"></div>
            
            <div className="demo-tabs">
              <button 
                className={`demo-tab ${activeTab === 'cnn' ? 'active' : ''}`}
                onClick={() => handleTabChange('cnn')}
              >
                🧠 CNN Tutorial
              </button>
              <button 
                className={`demo-tab ${activeTab === 'rag' ? 'active' : ''}`}
                onClick={() => handleTabChange('rag')}
              >
                📚 RAG Tutorial
              </button>
            </div>
            
            <div className="demo-content">
              {activeTab === 'cnn' && (
                <div className="demo-video cnn-demo">
                  <div className="video-placeholder">
                    <div className="video-icon">🎥</div>
                    <h4>CNN Building Tutorial</h4>
                    <p>Learn how to build convolutional neural networks step by step</p>
                    <ul>
                      <li>• Layer-by-layer architecture design</li>
                      <li>• Parameter optimization tips</li>
                      <li>• Pre-trained model integration</li>
                      <li>• Real-time code generation</li>
                    </ul>
                  </div>
                </div>
              )}
              
              {activeTab === 'rag' && (
                <div className="demo-video rag-demo">
                  <div className="video-placeholder">
                    <div className="video-icon">🎥</div>
                    <h4>RAG Pipeline Tutorial</h4>
                    <p>Master retrieval-augmented generation from scratch</p>
                    <ul>
                      <li>• Document processing & chunking</li>
                      <li>• Vector database setup</li>
                      <li>• Retrieval & reranking strategies</li>
                      <li>• LLM integration & prompting</li>
                    </ul>
                  </div>
                </div>
              )}
            </div>
            
            <button 
              className="btn-secondary demo-btn" 
              onClick={handleWatchDemo}
            >
              Try {activeTab === 'cnn' ? 'CNN' : 'RAG'} Builder
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;