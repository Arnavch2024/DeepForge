import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Cpu, TrendingUp, AlertCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const GPUEfficiencyWidget = ({ className = "" }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadGPUData();
  }, []);

  const loadGPUData = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:5000/api/visualizations/gpu_comparison');
      const result = await response.json();
      
      if (result.error) {
        throw new Error(result.error);
      }
      
      // Transform data for chart
      const transformedData = Object.entries(result.stats.CO2_kg).map(([gpu, co2]) => ({
        gpu,
        co2: parseFloat(co2.toFixed(2)),
        energy: parseFloat(result.stats.Energy_kWh[gpu].toFixed(2)),
        efficiency: parseFloat((co2 / result.stats.Training_Hours[gpu]).toFixed(2))
      }));
      
      setData(transformedData);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-8">
          <div className="text-center">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mx-auto mb-3"></div>
            <p className="text-sm text-gray-600">Loading GPU data...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-8">
          <div className="text-center">
            <AlertCircle className="h-6 w-6 text-red-500 mx-auto mb-3" />
            <p className="text-sm text-red-600">{error}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getEfficiencyLevel = (efficiency) => {
    if (efficiency < 0.5) return { level: 'Excellent', color: 'bg-green-100 text-green-800' };
    if (efficiency < 1.0) return { level: 'Good', color: 'bg-blue-100 text-blue-800' };
    if (efficiency < 1.5) return { level: 'Fair', color: 'bg-yellow-100 text-yellow-800' };
    return { level: 'Poor', color: 'bg-red-100 text-red-800' };
  };

  // Sort by efficiency (lower is better)
  const sortedData = [...data].sort((a, b) => a.efficiency - b.efficiency);

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Cpu className="h-5 w-5 text-blue-600" />
          <span>GPU Efficiency Ranking</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Efficiency Chart */}
        <div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={sortedData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="gpu" fontSize={10} />
              <YAxis fontSize={10} />
              <Tooltip 
                formatter={(value, name) => [
                  name === 'efficiency' ? `${value} kg/hr` : `${value} ${name === 'co2' ? 'kg' : 'kWh'}`,
                  name === 'efficiency' ? 'Efficiency Score' : name === 'co2' ? 'CO₂' : 'Energy'
                ]}
              />
              <Bar dataKey="efficiency" fill="#3b82f6" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* GPU Rankings */}
        <div className="space-y-2">
          <h4 className="font-semibold text-sm text-gray-900">Efficiency Rankings</h4>
          <div className="space-y-2">
            {sortedData.slice(0, 5).map((gpu, index) => {
              const efficiency = getEfficiencyLevel(gpu.efficiency);
              return (
                <div key={gpu.gpu} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                  <div className="flex items-center space-x-2">
                    <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                      index === 0 ? 'bg-yellow-400 text-yellow-900' :
                      index === 1 ? 'bg-gray-300 text-gray-700' :
                      index === 2 ? 'bg-orange-300 text-orange-900' :
                      'bg-gray-200 text-gray-600'
                    }`}>
                      {index + 1}
                    </div>
                    <div>
                      <div className="font-medium text-sm">{gpu.gpu}</div>
                      <div className="text-xs text-gray-600">
                        {gpu.co2} kg CO₂ • {gpu.energy} kWh
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <Badge className={`text-xs ${efficiency.color}`}>
                      {efficiency.level}
                    </Badge>
                    <div className="text-xs text-gray-600 mt-1">
                      {gpu.efficiency} kg/hr
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Best GPU Recommendation */}
        <div className="bg-green-50 p-3 rounded-lg">
          <div className="flex items-center space-x-2 mb-2">
            <TrendingUp className="h-4 w-4 text-green-600" />
            <span className="font-semibold text-sm text-green-900">Recommended</span>
          </div>
          <div className="text-sm text-green-800">
            <strong>{sortedData[0]?.gpu}</strong> offers the best efficiency with{' '}
            <strong>{sortedData[0]?.efficiency} kg CO₂/hour</strong> for training workloads.
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="p-2 bg-blue-50 rounded">
            <div className="text-lg font-bold text-blue-700">{data?.length || 0}</div>
            <div className="text-xs text-blue-600">GPU Types</div>
          </div>
          <div className="p-2 bg-green-50 rounded">
            <div className="text-lg font-bold text-green-700">
              {sortedData[0]?.efficiency.toFixed(2) || 0}
            </div>
            <div className="text-xs text-green-600">Best Score</div>
          </div>
          <div className="p-2 bg-yellow-50 rounded">
            <div className="text-lg font-bold text-yellow-700">
              {(data?.reduce((sum, gpu) => sum + gpu.co2, 0) / data?.length || 0).toFixed(1)}
            </div>
            <div className="text-xs text-yellow-600">Avg CO₂</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default GPUEfficiencyWidget;