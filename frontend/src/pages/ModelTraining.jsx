import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Brain, Zap, Leaf, Settings, Play, AlertCircle } from 'lucide-react';
import CarbonPredictionWidget from '../components/dashboard/CarbonPredictionWidget';

const ModelTraining = () => {
  const [trainingConfig, setTrainingConfig] = useState({
    modelType: 'transformer',
    parameters: '100000000',
    gpuType: 'A100',
    batchSize: '32',
    epochs: '10',
    learningRate: '0.001'
  });

  const [isTraining, setIsTraining] = useState(false);

  const handleConfigChange = (field, value) => {
    setTrainingConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const startTraining = () => {
    setIsTraining(true);
    // Simulate training process
    setTimeout(() => {
      setIsTraining(false);
    }, 5000);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="bg-blue-100 p-2 rounded-lg">
            <Brain className="h-6 w-6 text-blue-600" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Model Training</h1>
            <p className="text-gray-600">Configure and train your AI models with environmental awareness</p>
          </div>
        </div>
        <Badge variant="outline" className="text-green-600 border-green-300">
          <Leaf className="h-3 w-3 mr-1" />
          Carbon Optimized
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Training Configuration */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Settings className="h-5 w-5" />
                <span>Training Configuration</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="model" className="space-y-4">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="model">Model</TabsTrigger>
                  <TabsTrigger value="hardware">Hardware</TabsTrigger>
                  <TabsTrigger value="training">Training</TabsTrigger>
                </TabsList>

                <TabsContent value="model" className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Model Type</Label>
                      <Select value={trainingConfig.modelType} onValueChange={(value) => handleConfigChange('modelType', value)}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="transformer">Transformer</SelectItem>
                          <SelectItem value="cnn">CNN</SelectItem>
                          <SelectItem value="rnn">RNN</SelectItem>
                          <SelectItem value="bert">BERT</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Parameters</Label>
                      <Input
                        type="number"
                        value={trainingConfig.parameters}
                        onChange={(e) => handleConfigChange('parameters', e.target.value)}
                        placeholder="100000000"
                      />
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="hardware" className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label className="flex items-center space-x-1">
                        <Zap className="h-4 w-4" />
                        <span>GPU Type</span>
                      </Label>
                      <Select value={trainingConfig.gpuType} onValueChange={(value) => handleConfigChange('gpuType', value)}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="A100">NVIDIA A100 (High Performance)</SelectItem>
                          <SelectItem value="RTX 4090">RTX 4090 (Balanced)</SelectItem>
                          <SelectItem value="RTX 3060">RTX 3060 (Efficient)</SelectItem>
                          <SelectItem value="CPU">CPU (Low Power)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Batch Size</Label>
                      <Input
                        type="number"
                        value={trainingConfig.batchSize}
                        onChange={(e) => handleConfigChange('batchSize', e.target.value)}
                        placeholder="32"
                      />
                    </div>
                  </div>
                  
                  {/* Environmental Impact Warning */}
                  {trainingConfig.gpuType === 'A100' && (
                    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                      <div className="flex items-center space-x-2">
                        <AlertCircle className="h-4 w-4 text-yellow-600" />
                        <span className="text-sm font-medium text-yellow-800">High Carbon Impact</span>
                      </div>
                      <p className="text-sm text-yellow-700 mt-1">
                        A100 GPUs have high performance but also high energy consumption. Consider RTX 4090 for better efficiency.
                      </p>
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="training" className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Epochs</Label>
                      <Input
                        type="number"
                        value={trainingConfig.epochs}
                        onChange={(e) => handleConfigChange('epochs', e.target.value)}
                        placeholder="10"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Learning Rate</Label>
                      <Input
                        type="number"
                        step="0.0001"
                        value={trainingConfig.learningRate}
                        onChange={(e) => handleConfigChange('learningRate', e.target.value)}
                        placeholder="0.001"
                      />
                    </div>
                  </div>
                </TabsContent>
              </Tabs>

              <div className="mt-6 pt-4 border-t">
                <Button 
                  onClick={startTraining} 
                  disabled={isTraining}
                  className="w-full"
                  size="lg"
                >
                  {isTraining ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Training in Progress...
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Start Training
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Carbon Impact Prediction */}
        <div>
          <CarbonPredictionWidget />
        </div>
      </div>

      {/* Training Progress (shown when training) */}
      {isTraining && (
        <Card>
          <CardHeader>
            <CardTitle>Training Progress</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between text-sm">
                <span>Epoch 3/10</span>
                <span>30% Complete</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-blue-600 h-2 rounded-full w-[30%] transition-all duration-500"></div>
              </div>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Loss:</span>
                  <div className="font-medium">0.245</div>
                </div>
                <div>
                  <span className="text-gray-600">Accuracy:</span>
                  <div className="font-medium">87.3%</div>
                </div>
                <div>
                  <span className="text-gray-600">CO₂ So Far:</span>
                  <div className="font-medium text-green-600">1.2 kg</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ModelTraining;