import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Leaf, Zap, TrendingUp, AlertCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const CarbonImpactWidget = ({ className = "" }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadCarbonData();
  }, []);

  const loadCarbonData = async () => {
    try {
      setLoading(true);
      
      // Load dataset stats and scenarios
      const [statsResponse, scenariosResponse] = await Promise.all([
        fetch('http://localhost:5000/api/dataset/stats'),
        fetch('http://localhost:5000/api/scenarios')
      ]);
      
      const statsData = await statsResponse.json();
      const scenariosData = await scenariosResponse.json();
      
      if (statsData.error || scenariosData.error) {
        throw new Error(statsData.error || scenariosData.error);
      }
      
      setData({ stats: statsData, scenarios: scenariosData });
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
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading carbon impact data...</p>
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
            <AlertCircle className="h-8 w-8 text-red-500 mx-auto mb-4" />
            <p className="text-red-600 mb-4">{error}</p>
            <Button onClick={loadCarbonData} variant="outline" size="sm">
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  const { stats, scenarios } = data;
  
  // Transform scenarios for chart
  const scenarioData = Object.entries(scenarios.scenarios).map(([name, data]) => ({
    name: name.split(' - ')[0],
    co2: parseFloat(data.prediction.toFixed(2))
  }));

  const getImpactLevel = (co2) => {
    if (co2 < 1) return { level: 'Low', color: 'bg-green-100 text-green-800' };
    if (co2 < 3) return { level: 'Medium', color: 'bg-yellow-100 text-yellow-800' };
    if (co2 < 5) return { level: 'High', color: 'bg-orange-100 text-orange-800' };
    return { level: 'Very High', color: 'bg-red-100 text-red-800' };
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Leaf className="h-5 w-5 text-green-600" />
          <span>Carbon Impact Overview</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <Leaf className="h-6 w-6 text-green-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-green-700">
              {stats.co2_stats.mean.toFixed(2)} kg
            </div>
            <div className="text-sm text-green-600">Avg CO₂ per Training</div>
          </div>
          
          <div className="text-center p-4 bg-yellow-50 rounded-lg">
            <Zap className="h-6 w-6 text-yellow-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-yellow-700">
              {stats.energy_stats.mean.toFixed(2)} kWh
            </div>
            <div className="text-sm text-yellow-600">Avg Energy Usage</div>
          </div>
          
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <TrendingUp className="h-6 w-6 text-blue-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-blue-700">
              {stats.total_records.toLocaleString()}
            </div>
            <div className="text-sm text-blue-600">Training Records</div>
          </div>
        </div>

        {/* Scenario Comparison Chart */}
        <div>
          <h4 className="font-semibold text-gray-900 mb-3">Training Scenarios CO₂ Impact</h4>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={scenarioData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" fontSize={12} />
              <YAxis fontSize={12} />
              <Tooltip formatter={(value) => [`${value} kg`, 'CO₂ Emissions']} />
              <Bar dataKey="co2" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Quick Scenarios */}
        <div>
          <h4 className="font-semibold text-gray-900 mb-3">Quick Impact Assessment</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {scenarioData.slice(0, 4).map((scenario, index) => {
              const impact = getImpactLevel(scenario.co2);
              return (
                <div key={scenario.name} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div>
                    <div className="font-medium text-sm">{scenario.name}</div>
                    <div className="text-xs text-gray-600">{scenario.co2} kg CO₂</div>
                  </div>
                  <Badge className={`text-xs ${impact.color}`}>
                    {impact.level}
                  </Badge>
                </div>
              );
            })}
          </div>
        </div>

        {/* Environmental Context */}
        <div className="bg-blue-50 p-4 rounded-lg">
          <h4 className="font-semibold text-blue-900 mb-2">Environmental Impact</h4>
          <div className="text-sm text-blue-800 space-y-1">
            <div>• Average training = {(stats.co2_stats.mean * 2.2).toFixed(1)} miles driven</div>
            <div>• Total dataset impact = {(stats.co2_stats.mean * stats.total_records / 1000).toFixed(1)} tons CO₂</div>
            <div>• Equivalent to {(stats.energy_stats.mean * stats.total_records / 1000).toFixed(1)} MWh energy</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default CarbonImpactWidget;