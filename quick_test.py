import json

# Simple test to check if the backend is working
try:
    import requests
    
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
                {
                    "id": "e1-2",
                    "source": "1",
                    "target": "2"
                },
                {
                    "id": "e2-3",
                    "source": "2",
                    "target": "3"
                }
            ]
        }
    }
    
    response = requests.post(
        "http://localhost:8000/generate",
        json=test_data,
        timeout=10
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        if data.get("success") and data.get("code"):
            print("✅ Code generation successful!")
            print(f"Generated {len(data['code'])} characters of code")
            print("\n--- First 500 characters ---")
            print(data["code"][:500])
        else:
            print("❌ No code generated")
            print(f"Response: {data}")
    else:
        print(f"❌ Error: {response.text}")
        
except ImportError:
    print("❌ requests not available")
except Exception as e:
    print(f"❌ Error: {e}")