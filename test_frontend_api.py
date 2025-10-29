import requests
import json

def test_validation_endpoint():
    """Test the validation endpoint with the correct format"""
    test_data = {
        "builder_type": "cnn",
        "graph": {
            "nodes": [
                {
                    "id": "1",
                    "data": {
                        "type": "inputImage",
                        "params": {
                            "width": 32,
                            "height": 32,
                            "channels": 3
                        }
                    }
                }
            ],
            "edges": []
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/validate",
            json=test_data,
            timeout=5
        )
        
        print(f"Validation Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Validation Response: {data}")
            return True
        else:
            print(f"Validation Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Validation Exception: {e}")
        return False

def test_generate_endpoint():
    """Test the generate endpoint with a complete graph"""
    test_data = {
        "builder_type": "cnn",
        "graph": {
            "nodes": [
                {
                    "id": "1",
                    "data": {
                        "type": "dataset",
                        "params": {
                            "name": "CIFAR-10",
                            "train_split": 80,
                            "validation_split": 10,
                            "test_split": 10,
                            "batch_size": 32
                        }
                    }
                },
                {
                    "id": "2",
                    "data": {
                        "type": "inputImage",
                        "params": {
                            "width": 32,
                            "height": 32,
                            "channels": 3
                        }
                    }
                },
                {
                    "id": "3",
                    "data": {
                        "type": "output",
                        "params": {
                            "classes": 10,
                            "activation": "softmax"
                        }
                    }
                }
            ],
            "edges": [
                {"id": "e1-2", "source": "1", "target": "2"},
                {"id": "e2-3", "source": "2", "target": "3"}
            ]
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/generate",
            json=test_data,
            timeout=10
        )
        
        print(f"Generate Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("code"):
                print("✅ Code generation successful!")
                print(f"Generated {len(data['code'])} characters")
                return True
            else:
                print(f"❌ Generation failed: {data}")
                return False
        else:
            print(f"Generate Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Generate Exception: {e}")
        return False

if __name__ == "__main__":
    print("Testing DeepForge API endpoints...")
    print("\n1. Testing Validation Endpoint:")
    validation_ok = test_validation_endpoint()
    
    print("\n2. Testing Generate Endpoint:")
    generate_ok = test_generate_endpoint()
    
    print(f"\n=== Results ===")
    print(f"Validation: {'✅ PASS' if validation_ok else '❌ FAIL'}")
    print(f"Generation: {'✅ PASS' if generate_ok else '❌ FAIL'}")
    
    if validation_ok and generate_ok:
        print("\n🎉 All API endpoints working correctly!")
        print("The frontend should now work without 400 errors.")
    else:
        print("\n⚠️  Some endpoints still have issues.")