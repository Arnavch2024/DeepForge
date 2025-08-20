import React from 'react';
import './AdditionalFeatures.css';

const AdditionalFeatures = () => {
  const features = [
    {
      icon: '⚡',
      title: 'Real-Time Suggestions',
      description: 'Get instant code completions, bug fixes, and optimization tips as you type.'
    },
    {
      icon: '📖',
      title: 'Smart Documentation',
      description: 'AI retrieves relevant docs, papers, and examples based on your current context.'
    },
    {
      icon: '🎯',
      title: 'Learning-Focused',
      description: 'Understand the \'why\' behind every suggestion with detailed explanations.'
    },
    {
      icon: '🚀',
      title: 'Rapid Prototyping',
      description: 'Go from idea to working prototype in minutes with intelligent templates.'
    },
    {
      icon: '🔧',
      title: 'Performance Optimization',
      description: 'Automatic detection of bottlenecks with actionable improvement suggestions.'
    },
    {
      icon: '👥',
      title: 'For Everyone',
      description: 'From students learning AI to developers building production systems.'
    }
  ];

  return (
    <section className="additional-features">
      <div className="features-container">
        <h2 className="section-title">Powered by Intelligent Assistance</h2>
        <p className="section-subtitle">
          Every feature is designed to make AI development accessible, educational, and efficient.
        </p>
        
        <div className="features-list">
          {features.map((feature, index) => (
            <div key={index} className="feature-item fade-in">
              <div className="feature-icon">{feature.icon}</div>
              <h4 className="feature-title">{feature.title}</h4>
              <p className="feature-description">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default AdditionalFeatures;