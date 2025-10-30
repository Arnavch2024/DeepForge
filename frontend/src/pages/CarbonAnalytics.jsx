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
    gpuComparison: null,
    models: null
  });
  
  // Prediction form state
  const [predictionForm, setPredictionForm] = useState({
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
      
      const [statsRes, scenariosRes, gpuRes, modelsRes] = await Promise.all([
        fetch('http://localhost:5000/api/dataset/stats'),
        fetch('http://localhost:5000/api/scenarios'),
        fetch('http://localhost:5000/api/visualizations/gpu_comparison'),
        fetch('http://localhost:5000/api/models')
      ]);
      
      const [stats, scenarios, gpuComparison, models] = await Promise.all([
        statsRes.json(),
        scenariosRes.json(),
        gpuRes.json(),
        modelsRes.json()
      ]);
      
      setData({ stats, scenarios, gpuComparison, models });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    try {
      setPredicting(true);
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...predictionForm, model: 'linear_regression' })
      });
      
      const result = await response.json();
      if (result.error) throw new Error(result.error);
      
      setPrediction(result);
    } catch (err) {
      setError(err.message);
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

  const { stats, scenarios, gpuComparison, models } = data;

  // Transform data for charts
  const scenarioData = scenarios ? Object.entries(scenarios.scenarios).map(([name, data]) => ({
    name: name.split(' - ')[0],
    co2: parseFloat(data.prediction.toFixed(2))
  })) : [];

  const gpuData = gpuComparison ? Object.entries(gpuComparison.stats.CO2_kg).map(([gpu, co2]) => ({
    gpu,
    co2: parseFloat(co2.toFixed(2)),
    energy: parseFloat(gpuComparison.stats.Energy_kWh[gpu].toFixed(2))
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
                {models?.gpu_types?.map(gpu => (
                  <option key={gpu} value={gpu}>{gpu}</option>
                ))}
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
            disabled={predicting}
            style={{ width: '100%', marginBottom: '1.5rem' }}
          >
            {predicting ? '🔄 Predicting...' : '🌱 Predict CO₂ Impact'}
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
              <div style={{ marginBottom: '1rem', color: 'var(--text-muted)' }}>Predicted CO₂ Emissions</div>
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

        {/* GPU Efficiency */}
        <div className="hub-card" style={{ padding: '2rem' }}>
          <h3 style={{ 
            fontSize: '1.5rem', 
            fontWeight: 'bold', 
            marginBottom: '1.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            🖥️ GPU Efficiency Ranking
          </h3>
          
          {gpuData.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={gpuData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="gpu" fontSize={12} />
                <YAxis fontSize={12} />
                <Tooltip formatter={(value) => [`${value} kg`, 'CO₂ Emissions']} />
                <Bar dataKey="co2" fill="var(--primary-color)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Scenarios Comparison */}
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
            📈 Training Scenarios Comparison
          </h3>
          
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