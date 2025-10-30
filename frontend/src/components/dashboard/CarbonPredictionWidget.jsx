import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Calculator, Leaf, Loader2 } from 'lucide-react';

const CarbonPredictionWidget = ({ className = "" }) => {
  const [formData, setFormData] = useState({
    model_parameters: '100000000',
    gpu_type: 'A100',
    training_hours: '5.0',
    dataset_size_mb: '2500',
    energy_kwh: '4.0'
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [gpuTypes, setGpuTypes] = useState([]);

  useEffect(() => {
    loadGPUTypes();
  }, []);

  const loadGPUTypes = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/models');
      const data = await response.json();
      if (data.gpu_types) {
        setGpuTypes(data.gpu_types);
      }
    } catch (err) {
      console.error('Failed to load GPU types:', err);
    }
  };

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handlePredict = async () => {
    try {
      setLoading(true);
      
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...formData,
          model: 'random_forest'
        })
      });
      
      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      setPrediction(data);
    } catch (err) {
      console.error('Prediction failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const getImpactLevel = (co2) => {
    if (co2 < 1) return { level: 'Low', color: 'text-green-600', bg: 'bg-green-100' };
    if (co2 < 3) return { level: 'Medium', color: 'text-yellow-600', bg: 'bg-yellow-100' };
    if (co2 < 5) return { level: 'High', color: 'text-orange-600', bg: 'bg-orange-100' };
    return { level: 'Very High', color: 'text-red-600', bg: 'bg-red-100' };
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Calculator className="h-5 w-5 text-blue-600" />
          <span>CO₂ Impact Predictor</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Quick Input Form */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="model_parameters" className="text-xs">Model Parameters</Label>
            <Input
              id="model_parameters"
              type="number"
              value={formData.model_parameters}
              onChange={(e) => handleInputChange('model_parameters', e.target.value)}
              className="text-sm"
              placeholder="100000000"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="gpu_type" className="text-xs">GPU Type</Label>
            <Select value={formData.gpu_type} onValueChange={(value) => handleInputChange('gpu_type', value)}>
              <SelectTrigger className="text-sm">
                <SelectValue placeholder="Select GPU" />
              </SelectTrigger>
              <SelectContent>
                {gpuTypes.map((gpu) => (
                  <SelectItem key={gpu} value={gpu} className="text-sm">
                    {gpu}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="training_hours" className="text-xs">Training Hours</Label>
            <Input
              id="training_hours"
              type="number"
              step="0.1"
              value={formData.training_hours}
              onChange={(e) => handleInputChange('training_hours', e.target.value)}
              className="text-sm"
              placeholder="5.0"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="energy_kwh" className="text-xs">Energy (kWh)</Label>
            <Input
              id="energy_kwh"
              type="number"
              step="0.1"
              value={formData.energy_kwh}
              onChange={(e) => handleInputChange('energy_kwh', e.target.value)}
              className="text-sm"
              placeholder="4.0"
            />
          </div>
        </div>

        <Button 
          onClick={handlePredict} 
          disabled={loading}
          className="w-full"
          size="sm"
        >
          {loading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Predicting...
            </>
          ) : (
            <>
              <Leaf className="mr-2 h-4 w-4" />
              Predict CO₂ Impact
            </>
          )}
        </Button>

        {/* Prediction Result */}
        {prediction && (
          <div className="mt-4 p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg">
            <div className="text-center">
              <div className="text-3xl font-bold text-gray-900 mb-2">
                {prediction.prediction.toFixed(2)} kg
              </div>
              <div className="text-sm text-gray-600 mb-3">
                Predicted CO₂ Emissions
              </div>
              <div className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${getImpactLevel(prediction.prediction).bg} ${getImpactLevel(prediction.prediction).color}`}>
                {getImpactLevel(prediction.prediction).level} Impact
              </div>
            </div>
            
            <div className="mt-4 text-xs text-gray-600 space-y-1">
              <div>≈ {(prediction.prediction * 2.2).toFixed(1)} miles driven</div>
              <div>≈ {(prediction.prediction / 0.4).toFixed(1)} kWh electricity</div>
              <div>≈ {(prediction.prediction * 0.5).toFixed(1)} days phone charging</div>
            </div>
          </div>
        )}

        {!prediction && (
          <div className="text-center py-6 text-gray-500">
            <Leaf className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">Enter parameters to predict CO₂ impact</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default CarbonPredictionWidget;