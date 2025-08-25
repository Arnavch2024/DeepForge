import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import App from './App.jsx'
import BuilderHub from './pages/BuilderHub.jsx'
import CNNBuilder from './pages/CNNBuilder.jsx'
import RAGBuilder from './pages/RAGBuilder.jsx'
import Auth from './pages/Auth.jsx'
import NotFound from './pages/NotFound.jsx'
import Profile from './pages/Profile.jsx'
import ProtectedRoute from './components/auth/ProtectedRoute.jsx'
import './styles/global.css'
import Subscriptions from './pages/Subscriptions.jsx'
import Chats from './pages/Chats.jsx'
import Navbar from './components/layouts/navbar.jsx'

// Add this for the font
const link = document.createElement('link');
link.href = 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap';
link.rel = 'stylesheet';
document.head.appendChild(link);

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <div className="bg-grid" />
      <Navbar />
      <div className="app-shell">
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/builder" element={<ProtectedRoute><BuilderHub /></ProtectedRoute>} />
        <Route path="/builder/cnn" element={<ProtectedRoute><CNNBuilder /></ProtectedRoute>} />
        <Route path="/builder/rag" element={<ProtectedRoute><RAGBuilder /></ProtectedRoute>} />
        <Route path="/auth" element={<Auth />} />
        <Route path="/subscriptions" element={<ProtectedRoute><Subscriptions /></ProtectedRoute>} />
        <Route path="/chats" element={<ProtectedRoute><Chats /></ProtectedRoute>} />
        <Route path="/profile" element={<ProtectedRoute><Profile /></ProtectedRoute>} />
        <Route path="*" element={<NotFound />} />
      </Routes>
      </div>
    </BrowserRouter>
  </React.StrictMode>,
)