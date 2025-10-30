import React, { Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/layout/Sidebar';
import Dashboard from './pages/Dashboard';
import ModelTraining from './pages/ModelTraining';
import CarbonAnalytics from './pages/CarbonAnalytics';

// Lazy load other pages
const Datasets = React.lazy(() => import('./pages/Datasets'));
const GPUManagement = React.lazy(() => import('./pages/GPUManagement'));
const Analytics = React.lazy(() => import('./pages/Analytics'));
const Team = React.lazy(() => import('./pages/Team'));
const Documentation = React.lazy(() => import('./pages/Documentation'));
const Settings = React.lazy(() => import('./pages/Settings'));

const LoadingSpinner = () => (
  <div className="flex items-center justify-center h-64">
    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
  </div>
);

function App() {
  return (
    <Router>
      <div className="flex h-screen bg-gray-50">
        {/* Sidebar */}
        <Sidebar />
        
        {/* Main Content */}
        <div className="flex-1 overflow-auto">
          <div className="p-6">
            <Suspense fallback={<LoadingSpinner />}>
              <Routes>
                {/* Redirect root to dashboard */}
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                
                {/* Main pages */}
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/training" element={<ModelTraining />} />
                <Route path="/carbon-analytics" element={<CarbonAnalytics />} />
                
                {/* Lazy loaded pages */}
                <Route path="/datasets" element={<Datasets />} />
                <Route path="/gpu-management" element={<GPUManagement />} />
                <Route path="/analytics" element={<Analytics />} />
                <Route path="/team" element={<Team />} />
                <Route path="/docs" element={<Documentation />} />
                <Route path="/settings" element={<Settings />} />
                
                {/* Catch all route */}
                <Route path="*" element={<Navigate to="/dashboard" replace />} />
              </Routes>
            </Suspense>
          </div>
        </div>
      </div>
    </Router>
  );
}

export default App;