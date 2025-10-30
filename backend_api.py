"""
Carbon Impact Backend API
Flask API for serving model predictions and data to frontend
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent threading issues
import matplotlib.pyplot as plt
import seaborn as sns

# Configure matplotlib for server use
plt.ioff()  # Turn off interactive mode
matplotlib.rcParams['figure.max_open_warning'] = 0
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Global variables to store loaded models
models = {}
label_encoder = None
metadata = None
dataset = None

def load_models_and_data():
    """Load models, encoder, and dataset on startup"""
    global models, label_encoder, metadata, dataset
    
    try:
        # Load metadata
        with open('training_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load models
        for name, info in metadata['models'].items():
            model = joblib.load(info['filename'])
            models[name] = {
                'model': model,
                'mse': info['mse'],
                'r2': info['r2']
            }
        
        # Load label encoder
        label_encoder = joblib.load('gpu_label_encoder.pkl')
        
        # Load dataset
        dataset = pd.read_csv('synthetic_carbon_impact_dataset.csv')
        dataset['GPU_Type_Encoded'] = label_encoder.transform(dataset['GPU_Type'])
        
        print("✅ Successfully loaded models and data")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(models) > 0
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get information about available models"""
    if not models:
        return jsonify({'error': 'No models loaded'}), 500
    
    model_info = {}
    for name, info in models.items():
        model_info[name] = {
            'mse': info['mse'],
            'r2': info['r2']
        }
    
    return jsonify({
        'models': model_info,
        'gpu_types': label_encoder.classes_.tolist(),
        'training_date': metadata.get('training_date', 'Unknown')
    })

@app.route('/api/predict', methods=['POST'])
def predict_carbon_impact():
    """Predict CO2 emissions for given parameters"""
    try:
        data = request.get_json()
        
        # Extract parameters
        model_name = data.get('model', 'Enhanced Random Forest')  # Use best model as default
        model_params = float(data.get('model_parameters', 100000000))
        gpu_type = data.get('gpu_type', 'A100')
        training_hours = float(data.get('training_hours', 5.0))
        dataset_size_mb = float(data.get('dataset_size_mb', 2500))
        energy_kwh = float(data.get('energy_kwh', 4.0))
        
        # Validate model
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 400
        
        # Validate GPU type
        if gpu_type not in label_encoder.classes_:
            return jsonify({'error': f'Unknown GPU type: {gpu_type}'}), 400
        
        # Encode GPU type
        gpu_encoded = label_encoder.transform([gpu_type])[0]
        
        # Calculate engineered features (same as in training)
        params_per_hour = model_params / (training_hours + 0.1)
        energy_per_param = energy_kwh / (model_params / 1e6 + 0.1)
        gpu_power_factor = gpu_encoded * energy_kwh
        
        # Prepare features with engineered features
        features = [[
            model_params, gpu_encoded, training_hours, dataset_size_mb, energy_kwh,
            params_per_hour, energy_per_param, gpu_power_factor
        ]]
        
        # Make prediction
        model = models[model_name]['model']
        prediction = float(model.predict(features)[0])
        
        return jsonify({
            'prediction': prediction,
            'model_used': model_name,
            'model_r2': models[model_name]['r2'],
            'input_parameters': {
                'model_parameters': model_params,
                'gpu_type': gpu_type,
                'training_hours': training_hours,
                'dataset_size_mb': dataset_size_mb,
                'energy_kwh': energy_kwh
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset/stats', methods=['GET'])
def get_dataset_stats():
    """Get dataset statistics"""
    if dataset is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    stats = {
        'total_records': len(dataset),
        'gpu_types': dataset['GPU_Type'].value_counts().to_dict(),
        'co2_stats': {
            'mean': float(dataset['CO2_kg'].mean()),
            'min': float(dataset['CO2_kg'].min()),
            'max': float(dataset['CO2_kg'].max()),
            'std': float(dataset['CO2_kg'].std())
        },
        'energy_stats': {
            'mean': float(dataset['Energy_kWh'].mean()),
            'min': float(dataset['Energy_kWh'].min()),
            'max': float(dataset['Energy_kWh'].max()),
            'std': float(dataset['Energy_kWh'].std())
        },
        'training_hours_stats': {
            'mean': float(dataset['Training_Hours'].mean()),
            'min': float(dataset['Training_Hours'].min()),
            'max': float(dataset['Training_Hours'].max()),
            'std': float(dataset['Training_Hours'].std())
        }
    }
    
    return jsonify(stats)

@app.route('/api/visualizations/gpu_comparison', methods=['GET'])
def gpu_comparison_chart():
    """Generate GPU comparison chart as base64 image"""
    try:
        # Create GPU comparison chart
        plt.figure(figsize=(12, 6))
        
        gpu_stats = dataset.groupby('GPU_Type').agg({
            'CO2_kg': 'mean',
            'Energy_kWh': 'mean',
            'Training_Hours': 'mean'
        }).round(3)
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # CO2 emissions by GPU
        gpu_stats['CO2_kg'].plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Average CO2 Emissions by GPU Type')
        ax1.set_ylabel('CO2 (kg)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Energy consumption by GPU
        gpu_stats['Energy_kWh'].plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title('Average Energy Consumption by GPU Type')
        ax2.set_ylabel('Energy (kWh)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Proper cleanup
        plt.close('all')
        img_buffer.close()
        
        return jsonify({
            'image': img_base64,
            'stats': gpu_stats.to_dict()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/scenarios', methods=['GET'])
def get_prediction_scenarios():
    """Get predictions for predefined scenarios"""
    try:
        # Use best model
        best_model_name = max(models.keys(), key=lambda x: models[x]['r2'])
        model = models[best_model_name]['model']
        
        # Define scenarios with proper GPU encoding
        gpu_encodings = {gpu: idx for idx, gpu in enumerate(label_encoder.classes_)}
        
        scenarios = {
            'Small Model - CPU': {
                'base_params': [50000000, gpu_encodings['CPU'], 2.0, 1000, 0.13],
                'description': 'Small model on CPU'
            },
            'Medium Model - RTX 3060': {
                'base_params': [100000000, gpu_encodings['RTX 3060'], 5.0, 2500, 0.85],
                'description': 'Medium model on RTX 3060'
            },
            'Large Model - RTX 4090': {
                'base_params': [175000000, gpu_encodings['RTX 4090'], 8.0, 4000, 3.6],
                'description': 'Large model on RTX 4090'
            },
            'Large Model - A100': {
                'base_params': [175000000, gpu_encodings['A100'], 6.0, 4000, 2.4],
                'description': 'Large model on A100'
            },
            'Huge Model - A100': {
                'base_params': [300000000, gpu_encodings['A100'], 12.0, 5000, 4.8],
                'description': 'Huge model on A100'
            }
        }
        
        # Add engineered features to scenarios
        for name, scenario in scenarios.items():
            base = scenario['base_params']
            model_params, gpu_encoded, training_hours, dataset_size_mb, energy_kwh = base
            
            # Calculate engineered features
            params_per_hour = model_params / (training_hours + 0.1)
            energy_per_param = energy_kwh / (model_params / 1e6 + 0.1)
            gpu_power_factor = gpu_encoded * energy_kwh
            
            scenario['params'] = base + [params_per_hour, energy_per_param, gpu_power_factor]
        
        results = {}
        for name, scenario in scenarios.items():
            prediction = float(model.predict([scenario['params']])[0])
            results[name] = {
                'prediction': prediction,
                'description': scenario['description']
            }
        
        return jsonify({
            'scenarios': results,
            'model_used': best_model_name
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset/sample', methods=['GET'])
def get_dataset_sample():
    """Get a sample of the dataset"""
    try:
        sample_size = min(50, len(dataset))
        sample = dataset.sample(n=sample_size).to_dict('records')
        
        return jsonify({
            'sample': sample,
            'total_records': len(dataset),
            'sample_size': sample_size
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Carbon Impact Backend API...")
    
    # Load models and data
    if load_models_and_data():
        print("✅ Models and data loaded successfully")
        print("🌐 API endpoints available:")
        print("  - GET  /api/health - Health check")
        print("  - GET  /api/models - Model information")
        print("  - POST /api/predict - Make predictions")
        print("  - GET  /api/dataset/stats - Dataset statistics")
        print("  - GET  /api/visualizations/gpu_comparison - GPU comparison chart")
        print("  - GET  /api/scenarios - Prediction scenarios")
        print("  - GET  /api/dataset/sample - Dataset sample")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load models. Please run train_model.py first.")