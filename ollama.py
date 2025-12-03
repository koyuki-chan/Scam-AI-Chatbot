import requests
import json

def generate_with_ollama(prompt, model="qwen2:1.5b"):
    """
    用本地 Ollama 模型生成自然語氣回答
    """
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt}
    response = requests.post(url, json=payload, stream=True)
    output = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if "response" in data:
                output += data["response"]
    return output.strip()