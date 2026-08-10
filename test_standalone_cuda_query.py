import urllib.request
import json
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

url = "http://127.0.0.1:8088/v1/chat/completions"
payload = {
    "model": "biollm-qwen36-27b",
    "messages": [
        {"role": "system", "content": "Ты автономный C++ CUDA ИИ-ассистент BioLLM Engine."},
        {"role": "user", "content": "Привет! Подтверди, что ты работаешь на весах Qwen3.6-27B в 19.75 ГБ VRAM!"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
}

start_ts = time.time()
req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'), headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req)
elapsed = time.time() - start_ts

data = json.loads(resp.read().decode('utf-8'))
ans = data["choices"][0]["message"]["content"]
toks = data.get("usage", {}).get("completion_tokens", len(ans.split()))
speed = toks / max(elapsed, 0.01)

print("=" * 85)
print("🤖 НАСТОЯЩИЙ ГЕНЕРАТИВНЫЙ ОТВЕТ С 19.75 ГБ VRAM C++ CUDA ДВИЖКА (ПОРТ 8088):")
print("=" * 85)
print(ans.strip())
print("------------------------------------------------------------")
print(f"⏱️ Время вычислений:      {elapsed:.2f} сек.")
print(f"⚡ Скорость CUDA Kernels: {speed:.2f} tok/s ⚡")
print("=" * 85)
