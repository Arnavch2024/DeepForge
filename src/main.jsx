import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import App from './App.jsx'
import BuilderHub from './pages/BuilderHub.jsx'
import CNNBuilder from './pages/CNNBuilder.jsx'
import RAGBuilder from './pages/RAGBuilder.jsx'
import './styles/global.css'

// Add this for the font
const link = document.createElement('link');
link.href = 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap';
link.rel = 'stylesheet';
document.head.appendChild(link);

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/builder" element={<BuilderHub />} />
        <Route path="/builder/cnn" element={<CNNBuilder />} />
        <Route path="/builder/rag" element={<RAGBuilder />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
)