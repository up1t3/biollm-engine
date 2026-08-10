import urllib.request
import json

url = "http://127.0.0.1:8088/v1/chat/completions"

secret_key = "BIOLLM_SECRET_TOKEN_998231"
haystack = f"Параграф 0: Текст исследования систем хранения.\nПараграф 1: [СЕКРЕТНЫЙ КЛЮЧ ARCHITECTURE_KEY = '{secret_key}']\nПараграф 2: Продолжение исследования."

payload = {
    "model": "qwen3.6-27b-uncensored-hauhaucs-balanced",
    "messages": [
        {"role": "user", "content": f"Найди точное значение ARCHITECTURE_KEY в этом тексте и выведи только ключ:\n{haystack}"}
    ],
    "max_tokens": 300,
    "temperature": 0.1
}

req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
content = resp["choices"][0]["message"]["content"]

print("RAW RESPONSE FROM MODEL:")
print(content)
print("FOUND:", secret_key in content)
