import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api, clearAuthToken, getAuthToken } from '../../api/client.js';
import './navbar.css';

const Navbar = () => {
  const navigate = useNavigate();
  const [accountOpen, setAccountOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [theme, setTheme] = useState(localStorage.getItem('df_theme') || 'light');
  const [me, setMe] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        if (!getAuthToken()) return;
        const res = await api.me();
        setMe(res.user || res);
      } catch {}
    })();
  }, []);

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('theme-dark');
    } else {
      document.documentElement.classList.remove('theme-dark');
    }
    localStorage.setItem('df_theme', theme);
  }, [theme]);

  const onLogout = () => {
    clearAuthToken();
    navigate('/auth');
  };
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
          Deep Forge AI
        </div>
        
        <ul className="nav-links" style={{ display: menuOpen ? 'flex' : '' , flexDirection: menuOpen ? 'column' : 'row', gap: menuOpen ? '1rem' : '' }}>
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
            <Link className="nav-link" to="/builder">Builder</Link>
          </li>
          <li>
            <Link className="nav-link" to="/chats">Chats</Link>
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
          <li>
            <Link className="nav-link" to="/subscriptions">Subscriptions</Link>
          </li>
          {me && (
            <li>
              <Link className="nav-link" to="/profile">Profile</Link>
            </li>
          )}
        </ul>
        
        <button className="theme-btn" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label="Toggle theme">
          {theme === 'dark' ? '☀️' : '🌙'}
        </button>

        {me ? (
          <div style={{ position: 'relative' }}>
            <button className="account-btn" onClick={() => setAccountOpen(!accountOpen)}>
              {me.username || me.email}
            </button>
            {accountOpen && (
              <div className="account-dropdown">
                <div style={{ padding: 10, color: 'var(--text-muted)' }}>{me.email}</div>
                <Link to="/profile" className="nav-link" style={{ display: 'block', padding: 10 }}>Profile</Link>
                <Link to="/subscriptions" className="nav-link" style={{ display: 'block', padding: 10 }}>Subscriptions</Link>
                <Link to="/chats" className="nav-link" style={{ display: 'block', padding: 10 }}>Chats</Link>
                <button onClick={onLogout} className="nav-link" style={{ display: 'block', padding: 10, width: '100%', textAlign: 'left', background: 'transparent', border: 'none' }}>Logout</button>
              </div>
            )}
          </div>
        ) : (
          <Link className="signup-btn" to="/auth">Sign Up</Link>
        )}

        <button className="menu-btn" onClick={() => setMenuOpen(!menuOpen)} aria-label="Menu">☰</button>
      </div>
    </nav>
  );
};

export default Navbar;