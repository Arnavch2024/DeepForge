#!/usr/bin/env python3
"""
Quick test script to verify the Carbon Impact API is working correctly
"""
import requests
import json

def test_api():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Carbon Impact API...")
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check: {data['status']}, Models loaded: {data['models_loaded']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Get models
    try:
        response = requests.get(f"{base_url}/api/models")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models available: {list(data['models'].keys())}")
            print(f"✅ GPU types: {data['gpu_types']}")
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return False
    
    # Test 3: Make prediction with Random Forest
    try:
        prediction_data = {
            "model": "Random Forest",
            "model_parameters": 100000000,
            "gpu_type": "A100",
            "training_hours": 5.0,
            "dataset_size_mb": 2500,
            "energy_kwh": 4.0
        }
        
        response = requests.post(
            f"{base_url}/api/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(prediction_data)
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful: {data['prediction']:.2f} kg CO₂")
            print(f"✅ Model used: {data['model_used']}")
            print(f"✅ Model R²: {data['model_r2']:.4f}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False
    
    # Test 4: Get dataset stats
    try:
        response = requests.get(f"{base_url}/api/dataset/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Dataset stats: {data['total_records']} records")
            print(f"✅ Average CO₂: {data['co2_stats']['mean']:.2f} kg")
        else:
            print(f"❌ Dataset stats failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Dataset stats error: {e}")
        return False
    
    print("\n🎉 All API tests passed! The backend is working correctly.")
    return True

if __name__ == "__main__":
    test_api()