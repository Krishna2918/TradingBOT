import requests
import json

# Test Ollama API directly
url = "http://localhost:11434/api/generate"
payload = {
    "model": "qwen2.5:14b-instruct",
    "prompt": "Hello, test",
    "stream": False
}

try:
    response = requests.post(url, json=payload, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
