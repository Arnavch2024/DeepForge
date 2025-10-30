import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  Brain, 
  Database, 
  Zap, 
  Users, 
  TrendingUp, 
  Leaf,
  Clock,
  Cpu
} from 'lucide-react';
import CarbonImpactWidget from '../components/dashboard/CarbonImpactWidget';

const Dashboard = () => {
  const stats = [
    {
      title: 'Active Models',
      value: '12',
      change: '+2 this week',
      icon: Brain,
      color: 'blue'
    },
    {
      title: 'Datasets',
      value: '48',
      change: '+5 new',
      icon: Database,
      color: 'green'
    },
    {
      title: 'GPU Hours',
      value: '1,247',
      change: '87% efficiency',
      icon: Zap,
      color: 'yellow'
    },
    {
      title: 'Team Members',
      value: '8',
      change: '2 online',
      icon: Users,
      color: 'purple'
    }
  ];

  const recentActivity = [
    {
      id: 1,
      action: 'Model training completed',
      model: 'BERT-Large',
      time: '2 minutes ago',
      status: 'success',
      co2: '2.3 kg'
    },
    {
      id: 2,
      action: 'Dataset uploaded',
      model: 'Customer Reviews',
      time: '15 minutes ago',
      status: 'info',
      co2: null
    },
    {
      id: 3,
      action: 'GPU allocation optimized',
      model: 'RTX 4090 → A100',
      time: '1 hour ago',
      status: 'warning',
      co2: '-1.2 kg saved'
    },
    {
      id: 4,
      action: 'Model deployed',
      model: 'Sentiment Analysis v2',
      time: '3 hours ago',
      status: 'success',
      co2: null
    }
  ];

  const getStatusColor = (status) => {
    switch (status) {
      case 'success': return 'bg-green-100 text-green-800';
      case 'warning': return 'bg-yellow-100 text-yellow-800';
      case 'info': return 'bg-blue-100 text-blue-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getIconColor = (color) => {
    switch (color) {
      case 'blue': return 'text-blue-600 bg-blue-100';
      case 'green': return 'text-green-600 bg-green-100';
      case 'yellow': return 'text-yellow-600 bg-yellow-100';
      case 'purple': return 'text-purple-600 bg-purple-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600">Welcome back! Here's what's happening with your AI projects.</p>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant="outline" className="text-green-600 border-green-300">
            <Leaf className="h-3 w-3 mr-1" />
            Carbon Optimized
          </Badge>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <Card key={index}>
              <CardContent className="p-6">
                <div className="flex items-center space-x-4">
                  <div className={`p-3 rounded-lg ${getIconColor(stat.color)}`}>
                    <Icon className="h-6 w-6" />
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                    <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
                    <p className="text-xs text-gray-500">{stat.change}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Activity */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Clock className="h-5 w-5" />
                <span>Recent Activity</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentActivity.map((activity) => (
                  <div key={activity.id} className="flex items-center space-x-4 p-3 bg-gray-50 rounded-lg">
                    <div className={`w-2 h-2 rounded-full ${
                      activity.status === 'success' ? 'bg-green-500' :
                      activity.status === 'warning' ? 'bg-yellow-500' :
                      activity.status === 'info' ? 'bg-blue-500' : 'bg-gray-500'
                    }`}></div>
                    <div className="flex-1">
                      <p className="font-medium text-sm text-gray-900">{activity.action}</p>
                      <p className="text-sm text-gray-600">{activity.model}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-xs text-gray-500">{activity.time}</p>
                      {activity.co2 && (
                        <Badge className={`text-xs mt-1 ${
                          activity.co2.includes('saved') ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                        }`}>
                          {activity.co2}
                        </Badge>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Carbon Impact Widget */}
        <div>
          <CarbonImpactWidget />
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="cursor-pointer hover:shadow-md transition-shadow">
          <CardContent className="p-6 text-center">
            <Brain className="h-8 w-8 text-blue-600 mx-auto mb-3" />
            <h3 className="font-semibold text-gray-900 mb-2">Start Training</h3>
            <p className="text-sm text-gray-600">Begin training a new model with carbon optimization</p>
          </CardContent>
        </Card>

        <Card className="cursor-pointer hover:shadow-md transition-shadow">
          <CardContent className="p-6 text-center">
            <Database className="h-8 w-8 text-green-600 mx-auto mb-3" />
            <h3 className="font-semibold text-gray-900 mb-2">Upload Dataset</h3>
            <p className="text-sm text-gray-600">Add new training data to your collection</p>
          </CardContent>
        </Card>

        <Card className="cursor-pointer hover:shadow-md transition-shadow">
          <CardContent className="p-6 text-center">
            <Cpu className="h-8 w-8 text-purple-600 mx-auto mb-3" />
            <h3 className="font-semibold text-gray-900 mb-2">Optimize GPUs</h3>
            <p className="text-sm text-gray-600">Analyze and improve hardware efficiency</p>
          </CardContent>
        </Card>
      </div>

      {/* Performance Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <TrendingUp className="h-5 w-5" />
            <span>Performance Overview</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">94.2%</div>
              <div className="text-sm text-gray-600">Model Accuracy</div>
              <div className="text-xs text-green-600">+2.1% this week</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600 mb-2">87%</div>
              <div className="text-sm text-gray-600">GPU Efficiency</div>
              <div className="text-xs text-green-600">+5% optimized</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-yellow-600 mb-2">12.3</div>
              <div className="text-sm text-gray-600">Avg CO₂ (kg)</div>
              <div className="text-xs text-green-600">-15% reduced</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-600 mb-2">4.2h</div>
              <div className="text-sm text-gray-600">Avg Training Time</div>
              <div className="text-xs text-green-600">-20% faster</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Dashboard;