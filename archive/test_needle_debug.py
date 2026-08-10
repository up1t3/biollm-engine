import urllib.request
import json
import time

url = "http://127.0.0.1:8088/v1/chat/completions"

secret_key = "BIOLLM_SECRET_TOKEN_998231"
haystack = f"Вот большой текст исследования внимания. СЕКРЕТНЫЙ КЛЮЧ ARCHITECTURE_KEY = '{secret_key}'. Конец текста."

payload = {
    "model": "qwen3.6-27b-uncensored-hauhaucs-balanced",
    "messages": [
        {"role": "system", "content": "Отвечай строго и кратко без долгих размышлений."},
        {"role": "user", "content": f"Текст: {haystack}\n\nНапиши точное значение ARCHITECTURE_KEY."}
    ],
    "max_tokens": 300,
    "temperature": 0.0
}

req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
content = resp["choices"][0]["message"]["content"]

print("FULL CONTENT:")
print(content)
print("SECRET FOUND:", secret_key in content)
