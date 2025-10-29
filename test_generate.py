import requests
import json

# Test data for CNN generation
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
                        "width": 224,
                        "height": 224,
                        "channels": 3
                    }
                }
            },
            {
                "id": "3", 
                "data": {
                    "type": "conv2d",
                    "params": {
                        "filters": 32,
                        "kernel": "3x3",
                        "activation": "relu"
                    }
                }
            },
            {
                "id": "4",
                "data": {
                    "type": "flatten",
                    "params": {}
                }
            },
            {
                "id": "5",
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
            },
            {
                "id": "e3-4",
                "source": "3",
                "target": "4"
            },
            {
                "id": "e4-5",
                "source": "4",
                "target": "5"
            }
        ]
    }
}

try:
    response = requests.post(
        "http://localhost:8000/generate",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        data = response.json()
        if "code" in data:
            print("\n=== GENERATED CODE ===")
            print(data["code"])
        else:
            print("No code in response")
    
except Exception as e:
    print(f"Error: {e}")