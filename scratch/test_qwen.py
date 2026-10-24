import requests
import json
import time

url = "http://127.0.0.1:11434/api/generate"
prompt = 'Classify the column "MonthlyCharges" with sample values [29.85, 56.95, 53.85, 42.30] into a semantic domain and data type. Respond strictly as JSON: {"domain": "...", "data_type": "..."}'
payload = {
    "model": "qwen2.5:7b",
    "prompt": prompt,
    "format": "json",
    "stream": False
}
print("Querying qwen2.5:7b...")
t0 = time.time()
r = requests.post(url, json=payload, timeout=60)
elapsed = time.time() - t0
print(f"Elapsed: {elapsed:.2f}s")
if r.status_code == 200:
    print("Response:", r.json().get("response"))
else:
    print("Error:", r.status_code, r.text)
