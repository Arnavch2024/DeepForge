import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import BackButton from '../components/BackButton';
import '../styles/builder.css';

const CarbonAnalytics = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [data, setData] = useState({
    stats: null,
    scenarios: null,
    gpuEfficiency: null,
    modelSizeImpact: null,
    gpuComparison: null,
    models: null
  });
  
  // Prediction form state
  const [predictionForm, setPredictionForm] = useState({
    model: 'Enhanced Random Forest',
    model_parameters: '100000000',
    gpu_type: 'A100',
    training_hours: '5.0',
    dataset_size_mb: '2500',
    energy_kwh: '4.0'
  });
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);

  useEffect(() => {
    loadAllData();
  }, []);

  const loadAllData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Check if API is available first
      const healthRes = await fetch('http://localhost:5000/api/health');
      if (!healthRes.ok) {
        throw new Error('Carbon Impact API is not available. Please make sure the backend server is running on port 5000.');
      }
      
      const [statsRes, scenariosRes, gpuEffRes, modelSizeRes, gpuCompRes, modelsRes] = await Promise.all([
        fetch('http://localhost:5000/api/dataset/stats'),
        fetch('http://localhost:5000/api/scenarios'),
        fetch('http://localhost:5000/api/visualizations/gpu_efficiency'),
        fetch('http://localhost:5000/api/visualizations/model_size_impact'),
        fetch('http://localhost:5000/api/gpu_hardware_comparison'),
        fetch('http://localhost:5000/api/models')
      ]);
      
      // Check if all responses are ok
      if (!statsRes.ok || !scenariosRes.ok || !gpuEffRes.ok || !modelSizeRes.ok || !gpuCompRes.ok || !modelsRes.ok) {
        throw new Error('Failed to load carbon impact data. Please check if the backend API is running properly.');
      }
      
      const [stats, scenarios, gpuEfficiency, modelSizeImpact, gpuComparison, models] = await Promise.all([
        statsRes.json(),
        scenariosRes.json(),
        gpuEffRes.json(),
        modelSizeRes.json(),
        gpuCompRes.json(),
        modelsRes.json()
      ]);
      
      // Check for API errors in responses
      if (stats.error || scenarios.error || gpuEfficiency.error || modelSizeImpact.error || gpuComparison.error || models.error) {
        throw new Error(stats.error || scenarios.error || gpuEfficiency.error || modelSizeImpact.error || gpuComparison.error || models.error);
      }
      
      setData({ stats, scenarios, gpuEfficiency, modelSizeImpact, gpuComparison, models });
    } catch (err) {
      console.error('Carbon Analytics Error:', err);
      setError(err.message || 'Failed to load carbon impact data');
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    try {
      setPredicting(true);
      setError(null);
      
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          ...predictionForm,
          model: predictionForm.model || 'Enhanced Random Forest'
        })
      });
      
      if (!response.ok) {
        if (response.status === 400) {
          throw new Error('Invalid prediction parameters. Please check your input values.');
        } else if (response.status === 500) {
          throw new Error('Server error during prediction. Please try again.');
        } else {
          throw new Error(`Prediction failed with status ${response.status}`);
        }
      }
      
      const result = await response.json();
      if (result.error) throw new Error(result.error);
      
      setPrediction(result);
    } catch (err) {
      console.error('Prediction Error:', err);
      setError(`Prediction failed: ${err.message}`);
    } finally {
      setPredicting(false);
    }
  };

  const getImpactLevel = (co2) => {
    if (co2 < 1) return { level: 'Low', color: 'bg-green-100 text-green-800' };
    if (co2 < 3) return { level: 'Medium', color: 'bg-yellow-100 text-yellow-800' };
    if (co2 < 5) return { level: 'High', color: 'bg-orange-100 text-orange-800' };
    return { level: 'Very High', color: 'bg-red-100 text-red-800' };
  };

  if (loading) {
    return (
      <div className="builder-hub">
        <BackButton />
        <div style={{ textAlign: 'center', padding: '4rem 0' }}>
          <div style={{ 
            width: '40px', 
            height: '40px', 
            border: '4px solid var(--border-color)', 
            borderTop: '4px solid var(--primary-color)', 
            borderRadius: '50%', 
            animation: 'spin 1s linear infinite',
            margin: '0 auto 1rem'
          }}></div>
          <p style={{ color: 'var(--text-muted)' }}>Loading Carbon Analytics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="builder-hub">
        <BackButton />
        <div style={{ textAlign: 'center', padding: '4rem 0' }}>
          <div style={{ 
            background: 'var(--bg-card)', 
            padding: '2rem', 
            borderRadius: '1rem', 
            boxShadow: 'var(--shadow-medium)',
            maxWidth: '400px',
            margin: '0 auto'
          }}>
            <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>⚠️</div>
            <p style={{ color: 'var(--text-primary)', marginBottom: '1rem' }}>{error}</p>
            <button className="btn-primary" onClick={loadAllData}>Retry</button>
          </div>
        </div>
      </div>
    );
  }

  const { stats, scenarios, gpuEfficiency, modelSizeImpact, gpuComparison, models } = data;

  // Transform data for charts
  const scenarioData = scenarios ? Object.entries(scenarios.scenarios).map(([name, data]) => ({
    name: name.replace(' Model', '').replace('(', '').replace(')', ''),
    co2: parseFloat(data.prediction.toFixed(2))
  })) : [];

  const gpuEfficiencyData = gpuEfficiency ? Object.entries(gpuEfficiency.stats.CO2_kg).map(([gpu, co2]) => ({
    gpu,
    co2: parseFloat(co2.toFixed(2)),
    energy: parseFloat(gpuEfficiency.stats.Energy_kWh[gpu].toFixed(2))
  })) : [];

  const modelSizeData = modelSizeImpact ? Object.entries(modelSizeImpact.stats.CO2_kg).map(([size, co2]) => ({
    size,
    co2: parseFloat(co2.toFixed(2)),
    energy: parseFloat(modelSizeImpact.stats.Energy_kWh[size].toFixed(2))
  })) : [];

  const gpuHardwareData = gpuComparison ? Object.entries(gpuComparison.gpu_comparison).map(([gpu, data]) => ({
    gpu,
    co2: parseFloat(data.prediction.toFixed(2)),
    energy: parseFloat(data.energy_kwh.toFixed(2)),
    hours: parseFloat(data.training_hours.toFixed(1))
  })) : [];

  return (
    <div className="builder-hub">
      <BackButton />
      
      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
        <div style={{ 
          fontSize: '3rem', 
          marginBottom: '1rem',
          background: 'var(--gradient-primary)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text'
        }}>
          🌱
        </div>
        <h1 className="hub-title">Carbon Impact Analytics</h1>
        <p style={{ 
          color: 'var(--text-muted)', 
          fontSize: '1.1rem', 
          maxWidth: '600px', 
          margin: '0 auto' 
        }}>
          Monitor and optimize the environmental impact of your ML training
        </p>
      </div>

      {/* API Status */}
      {error && (
        <div style={{ 
          background: '#fef3c7', 
          border: '2px solid #f59e0b', 
          borderRadius: '1rem', 
          padding: '1rem', 
          marginBottom: '2rem',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>⚠️</div>
          <div style={{ color: '#92400e', fontWeight: 'bold', marginBottom: '0.5rem' }}>
            Carbon Impact API Unavailable
          </div>
          <div style={{ color: '#92400e', fontSize: '0.9rem' }}>
            Make sure the backend server is running: <code>python backend_api.py</code>
          </div>
        </div>
      )}

      {/* Key Metrics */}
      {stats && (
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', 
          gap: '1.5rem', 
          marginBottom: '3rem' 
        }}>
          <div className="hub-card" style={{ textAlign: 'center', padding: '2rem' }}>
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🌱</div>
            <div style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--primary-color)', marginBottom: '0.5rem' }}>
              {stats.co2_stats.mean.toFixed(2)} kg
            </div>
            <div style={{ color: 'var(--text-muted)' }}>Avg CO₂ Emissions</div>
          </div>

          <div className="hub-card" style={{ textAlign: 'center', padding: '2rem' }}>
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>⚡</div>
            <div style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--primary-color)', marginBottom: '0.5rem' }}>
              {stats.energy_stats.mean.toFixed(2)} kWh
            </div>
            <div style={{ color: 'var(--text-muted)' }}>Avg Energy Usage</div>
          </div>

          <div className="hub-card" style={{ textAlign: 'center', padding: '2rem' }}>
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📊</div>
            <div style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--primary-color)', marginBottom: '0.5rem' }}>
              {stats.total_records.toLocaleString()}
            </div>
            <div style={{ color: 'var(--text-muted)' }}>Training Records</div>
          </div>

          <div className="hub-card" style={{ textAlign: 'center', padding: '2rem' }}>
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🖥️</div>
            <div style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--primary-color)', marginBottom: '0.5rem' }}>
              {Object.keys(stats.gpu_types).length}
            </div>
            <div style={{ color: 'var(--text-muted)' }}>GPU Types</div>
          </div>
        </div>
      )}

      {/* Main Content Grid */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', 
        gap: '2rem', 
        marginBottom: '3rem' 
      }}>
        {/* CO₂ Prediction */}
        <div className="hub-card" style={{ padding: '2rem' }}>
          <h3 style={{ 
            fontSize: '1.5rem', 
            fontWeight: 'bold', 
            marginBottom: '1.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            🧮 CO₂ Impact Predictor
          </h3>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>ML Model</label>
              <select
                value={predictionForm.model || 'Enhanced Random Forest'}
                onChange={(e) => setPredictionForm(prev => ({...prev, model: e.target.value}))}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  border: '2px solid var(--border-color)',
                  borderRadius: '0.5rem',
                  background: 'var(--bg-primary)',
                  color: 'var(--text-primary)'
                }}
              >
                {models?.models ? Object.keys(models.models).map(modelName => (
                  <option key={modelName} value={modelName}>
                    {modelName} (R² = {models.models[modelName].r2.toFixed(3)})
                  </option>
                )) : (
                  <>
                    <option value="Enhanced Random Forest">Enhanced Random Forest</option>
                    <option value="Random Forest">Random Forest</option>
                    <option value="Linear Regression">Linear Regression</option>
                  </>
                )}
              </select>
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>Model Parameters</label>
              <input
                type="number"
                value={predictionForm.model_parameters}
                onChange={(e) => setPredictionForm(prev => ({...prev, model_parameters: e.target.value}))}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  border: '2px solid var(--border-color)',
                  borderRadius: '0.5rem',
                  background: 'var(--bg-primary)',
                  color: 'var(--text-primary)'
                }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>GPU Type</label>
              <select
                value={predictionForm.gpu_type}
                onChange={(e) => setPredictionForm(prev => ({...prev, gpu_type: e.target.value}))}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  border: '2px solid var(--border-color)',
                  borderRadius: '0.5rem',
                  background: 'var(--bg-primary)',
                  color: 'var(--text-primary)'
                }}
              >
                {models?.gpu_types ? models.gpu_types.map(gpu => (
                  <option key={gpu} value={gpu}>{gpu}</option>
                )) : (
                  <>
                    <option value="A100">A100</option>
                    <option value="RTX 4090">RTX 4090</option>
                    <option value="RTX 4050">RTX 4050</option>
                    <option value="RTX 3060">RTX 3060</option>
                    <option value="RTX 3050">RTX 3050</option>
                    <option value="TPU">TPU</option>
                    <option value="CPU">CPU</option>
                  </>
                )}
              </select>
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>Training Hours</label>
              <input
                type="number"
                step="0.1"
                value={predictionForm.training_hours}
                onChange={(e) => setPredictionForm(prev => ({...prev, training_hours: e.target.value}))}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  border: '2px solid var(--border-color)',
                  borderRadius: '0.5rem',
                  background: 'var(--bg-primary)',
                  color: 'var(--text-primary)'
                }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>Energy (kWh)</label>
              <input
                type="number"
                step="0.1"
                value={predictionForm.energy_kwh}
                onChange={(e) => setPredictionForm(prev => ({...prev, energy_kwh: e.target.value}))}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  border: '2px solid var(--border-color)',
                  borderRadius: '0.5rem',
                  background: 'var(--bg-primary)',
                  color: 'var(--text-primary)'
                }}
              />
            </div>
          </div>

          <button 
            className="btn-primary" 
            onClick={handlePredict} 
            disabled={predicting || !data.models}
            style={{ width: '100%', marginBottom: '1.5rem' }}
          >
            {predicting ? '🔄 Predicting...' : 
             !data.models ? '⚠️ API Required for Predictions' : 
             '🌱 Predict CO₂ Impact (Random Forest Model)'}
          </button>

          {prediction && (
            <div style={{
              background: 'var(--gradient-secondary)',
              padding: '1.5rem',
              borderRadius: '1rem',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '2.5rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                {prediction.prediction.toFixed(2)} kg
              </div>
              <div style={{ marginBottom: '1rem', color: 'var(--text-muted)' }}>
                Predicted CO₂ Emissions (Random Forest Model)
              </div>
              <div style={{
                display: 'inline-block',
                padding: '0.5rem 1rem',
                borderRadius: '2rem',
                background: getImpactLevel(prediction.prediction).level === 'Low' ? '#dcfce7' : 
                           getImpactLevel(prediction.prediction).level === 'Medium' ? '#fef3c7' : '#fecaca',
                color: getImpactLevel(prediction.prediction).level === 'Low' ? '#166534' : 
                       getImpactLevel(prediction.prediction).level === 'Medium' ? '#92400e' : '#991b1b',
                fontWeight: 'bold',
                fontSize: '0.875rem'
              }}>
                {getImpactLevel(prediction.prediction).level} Impact
              </div>
              <div style={{ marginTop: '1rem', fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                <div>≈ {(prediction.prediction * 2.2).toFixed(1)} miles driven</div>
                <div>≈ {(prediction.prediction / 0.4).toFixed(1)} kWh electricity</div>
              </div>
            </div>
          )}
        </div>

        {/* Model Size Impact */}
        <div className="hub-card" style={{ padding: '2rem' }}>
          <h3 style={{ 
            fontSize: '1.5rem', 
            fontWeight: 'bold', 
            marginBottom: '1.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            📊 Model Size Impact
          </h3>
          <p style={{ color: 'var(--text-muted)', marginBottom: '1rem', fontSize: '0.9rem' }}>
            How model complexity affects carbon emissions
          </p>
          
          {modelSizeData.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={modelSizeData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="size" fontSize={12} />
                <YAxis fontSize={12} />
                <Tooltip formatter={(value) => [`${value} kg`, 'CO₂ Emissions']} />
                <Bar dataKey="co2" fill="var(--primary-color)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Model Size Scenarios Comparison */}
      {scenarioData.length > 0 && (
        <div className="hub-card" style={{ padding: '2rem', marginBottom: '2rem' }}>
          <h3 style={{ 
            fontSize: '1.5rem', 
            fontWeight: 'bold', 
            marginBottom: '1.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            📈 Model Size Scenarios
          </h3>
          <p style={{ color: 'var(--text-muted)', marginBottom: '1rem', fontSize: '0.9rem' }}>
            CO₂ impact of different model sizes (using optimal GPU for each)
          </p>
          
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={scenarioData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip formatter={(value) => [`${value} kg`, 'CO₂ Emissions']} />
              <Bar dataKey="co2" fill="var(--primary-color)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* GPU Comparisons Grid */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', 
        gap: '2rem', 
        marginBottom: '2rem' 
      }}>
        {/* GPU Efficiency (Same Workload) */}
        <div className="hub-card" style={{ padding: '2rem' }}>
          <h3 style={{ 
            fontSize: '1.5rem', 
            fontWeight: 'bold', 
            marginBottom: '1rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            🖥️ GPU Efficiency
          </h3>
          <p style={{ color: 'var(--text-muted)', marginBottom: '1rem', fontSize: '0.9rem' }}>
            Same model size (~100M params) on different GPUs
          </p>
          
          {gpuEfficiencyData.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={gpuEfficiencyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="gpu" fontSize={12} />
                <YAxis fontSize={12} />
                <Tooltip formatter={(value) => [`${value} kg`, 'CO₂ Emissions']} />
                <Bar dataKey="co2" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* GPU Hardware Comparison */}
        <div className="hub-card" style={{ padding: '2rem' }}>
          <h3 style={{ 
            fontSize: '1.5rem', 
            fontWeight: 'bold', 
            marginBottom: '1rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            ⚡ Hardware Performance
          </h3>
          <p style={{ color: 'var(--text-muted)', marginBottom: '1rem', fontSize: '0.9rem' }}>
            100M parameter model on different hardware
          </p>
          
          {gpuHardwareData.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={gpuHardwareData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="gpu" fontSize={12} />
                <YAxis fontSize={12} />
                <Tooltip formatter={(value) => [`${value} kg`, 'CO₂ Emissions']} />
                <Bar dataKey="co2" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Environmental Impact Summary */}
      {stats && (
        <div className="hub-card" style={{ padding: '2rem' }}>
          <h3 style={{ 
            fontSize: '1.5rem', 
            fontWeight: 'bold', 
            marginBottom: '1.5rem',
            textAlign: 'center'
          }}>
            🌍 Environmental Impact Summary
          </h3>
          
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
            gap: '1.5rem' 
          }}>
            <div style={{ 
              textAlign: 'center', 
              padding: '1.5rem', 
              background: 'var(--gradient-secondary)', 
              borderRadius: '1rem' 
            }}>
              <div style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                {(stats.co2_stats.mean * stats.total_records / 1000).toFixed(1)} tons
              </div>
              <div style={{ color: 'var(--text-muted)' }}>Total CO₂ Impact</div>
            </div>
            
            <div style={{ 
              textAlign: 'center', 
              padding: '1.5rem', 
              background: 'var(--gradient-accent)', 
              borderRadius: '1rem' 
            }}>
              <div style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                {(stats.energy_stats.mean * stats.total_records / 1000).toFixed(1)} MWh
              </div>
              <div style={{ color: 'var(--text-muted)' }}>Total Energy Consumed</div>
            </div>
            
            <div style={{ 
              textAlign: 'center', 
              padding: '1.5rem', 
              background: 'var(--gradient-primary)', 
              borderRadius: '1rem' 
            }}>
              <div style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                {((stats.co2_stats.mean * stats.total_records) * 2.2 / 1000).toFixed(1)}k
              </div>
              <div style={{ color: 'var(--text-muted)' }}>Miles Driven Equivalent</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CarbonAnalytics;