import React from 'react';
import { Link } from 'react-router-dom';
import './navbar.css';

const Navbar = () => {
  const handleSmoothScroll = (e, targetId) => {
    e.preventDefault();
    const target = document.querySelector(targetId);
    if (target) {
      target.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  };

  return (
    <nav className="navbar">
      <div className="nav-container">
        <div 
          className="logo" 
          onClick={(e) => handleSmoothScroll(e, '#home')}
        >
          AI Copilot
        </div>
        
        <ul className="nav-links">
          <li>
            <a 
              className="nav-link" 
              onClick={(e) => handleSmoothScroll(e, '#home')}
            >
              Home
            </a>
          </li>
          <li>
            <a 
              className="nav-link" 
              onClick={(e) => handleSmoothScroll(e, '#features')}
            >
              Features
            </a>
          </li>
          <li>
            <a className="nav-link" href="#docs">
              Docs
            </a>
          </li>
          <li>
            <a className="nav-link" href="#pricing">
              Pricing
            </a>
          </li>
          <li>
            <Link className="nav-link" to="/builder/cnn">
              CNN Builder
            </Link>
          </li>
          <li>
            <Link className="nav-link" to="/builder/rag">
              RAG Builder
            </Link>
          </li>
        </ul>
        
        <button 
          className="signup-btn"
          onClick={() => alert('Sign up clicked!')}
        >
          Sign Up
        </button>
      </div>
    </nav>
  );
};

export default Navbar;