import requests
import json

# Simple RAG test with minimal components
test_data = {
    "builder_type": "rag",
    "graph": {
        "nodes": [
            {
                "id": "1",
                "data": {
                    "type": "inputDocs",
                    "params": {
                        "source": "upload",
                        "path": "/docs"
                    }
                }
            },
            {
                "id": "2",
                "data": {
                    "type": "llm",
                    "params": {
                        "model": "gpt-4o-mini",
                        "temperature": 0.7
                    }
                }
            },
            {
                "id": "3",
                "data": {
                    "type": "output",
                    "params": {}
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
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        if data.get("success") and data.get("code"):
            print("✅ RAG code generation working!")
            print(f"Generated {len(data['code'])} characters")
            print("\n--- Generated Code Preview ---")
            print(data["code"][:300])
        else:
            print("❌ No code generated")
            print(f"Response: {data}")
    else:
        print(f"❌ Error: {response.text}")
        
except Exception as e:
    print(f"❌ Error: {e}")