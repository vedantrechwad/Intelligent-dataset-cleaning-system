import requests
import json
import time

url = "http://127.0.0.1:11434/api/generate"
prompt = 'Classify the column "MonthlyCharges" with sample values [29.85, 56.95, 53.85, 42.30] into a semantic domain and data type. Respond strictly as JSON: {"domain": "...", "data_type": "..."}'

for model in ["qwen2.5:3b", "qwen2.5:7b"]:
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False
    }
    print(f"\nTesting {model}...")
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=60)
    elapsed = time.time() - t0
    if r.status_code == 200:
        resp = r.json()
        print(f"  {model} Time: {elapsed:.2f}s")
        print(f"  {model} Response: {resp.get('response')}")
        eval_count = resp.get("eval_count", 0)
        eval_duration_s = resp.get("eval_duration", 1) / 1e9
        if eval_duration_s > 0:
            print(f"  {model} Generation Speed: {eval_count / eval_duration_s:.1f} tokens/sec")
    else:
        print(f"  {model} Error:", r.status_code, r.text)
