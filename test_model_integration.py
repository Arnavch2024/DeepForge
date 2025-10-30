"""
Test Model Integration
Quick test to verify model names and predictions work correctly
"""
import requests
import json

def test_model_integration():
    """Test the model integration between frontend and backend"""
    
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Model Integration...")
    
    # Test 1: Check available models
    print("\n1️⃣ Testing /api/models endpoint...")
    try:
        response = requests.get(f"{base_url}/api/models")
        if response.status_code == 200:
            models_data = response.json()
            print("✅ Models endpoint working!")
            print(f"Available models: {list(models_data['models'].keys())}")
            print(f"GPU types: {models_data['gpu_types']}")
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return False
    
    # Test 2: Test prediction with each model
    print("\n2️⃣ Testing predictions with different models...")
    
    test_params = {
        "model_parameters": 100000000,
        "gpu_type": "RTX 4090",
        "training_hours": 8.0,
        "dataset_size_mb": 3000,
        "energy_kwh": 3.6
    }
    
    for model_name in models_data['models'].keys():
        try:
            test_data = {**test_params, "model": model_name}
            response = requests.post(f"{base_url}/api/predict", json=test_data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {model_name}: {result['prediction']:.3f} kg CO2 (R² = {result['model_r2']:.3f})")
            else:
                print(f"❌ {model_name}: Failed with status {response.status_code}")
                
        except Exception as e:
            print(f"❌ {model_name}: Error - {e}")
    
    # Test 3: Test GPU differentiation
    print("\n3️⃣ Testing GPU differentiation...")
    
    gpu_test_params = {
        "model": "Enhanced Random Forest",
        "model_parameters": 150000000,
        "training_hours": 6.0,
        "dataset_size_mb": 2500,
        "energy_kwh": 2.4
    }
    
    gpu_results = {}
    for gpu_type in models_data['gpu_types']:
        try:
            test_data = {**gpu_test_params, "gpu_type": gpu_type}
            response = requests.post(f"{base_url}/api/predict", json=test_data)
            
            if response.status_code == 200:
                result = response.json()
                gpu_results[gpu_type] = result['prediction']
                print(f"✅ {gpu_type}: {result['prediction']:.3f} kg CO2")
            else:
                print(f"❌ {gpu_type}: Failed")
                
        except Exception as e:
            print(f"❌ {gpu_type}: Error - {e}")
    
    # Check if GPU values are different
    if len(set(round(v, 2) for v in gpu_results.values())) > 1:
        print("✅ GPU differentiation working - different carbon values!")
        
        # Show the range
        min_gpu = min(gpu_results, key=gpu_results.get)
        max_gpu = max(gpu_results, key=gpu_results.get)
        print(f"📊 Range: {min_gpu} ({gpu_results[min_gpu]:.3f}) → {max_gpu} ({gpu_results[max_gpu]:.3f})")
        print(f"🔥 Difference: {((gpu_results[max_gpu] / gpu_results[min_gpu]) - 1) * 100:.1f}% increase")
    else:
        print("❌ GPU differentiation not working - all values are the same!")
        return False
    
    print("\n🎉 Model integration test completed successfully!")
    return True

if __name__ == "__main__":
    test_model_integration()