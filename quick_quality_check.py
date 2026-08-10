import urllib.request
import json
import ast
import re

url = "http://127.0.0.1:8088/v1/chat/completions"

# 1. AST Code Test
payload_code = {
    "model": "qwen3.6-27b-uncensored-hauhaucs-balanced",
    "messages": [
        {"role": "system", "content": "Напиши только код на Python в блоке ```python."},
        {"role": "user", "content": "Напиши функцию `binary_search(arr, target)` с типизацией type hinting."}
    ],
    "max_tokens": 200,
    "temperature": 0.1
}

req1 = urllib.request.Request(url, data=json.dumps(payload_code).encode('utf-8'), headers={"Content-Type": "application/json"})
resp1 = json.loads(urllib.request.urlopen(req1).read().decode('utf-8'))
content1 = resp1["choices"][0]["message"]["content"]

if "<think>" in content1 and "</think>" in content1:
    content1 = content1.split("</think>")[-1]

code_clean = content1.split("```python")[1].split("```")[0].strip() if "```python" in content1 else content1.strip()

try:
    ast.parse(code_clean)
    ast_ok = True
except Exception:
    ast_ok = False

# 2. Strict JSON Test
payload_json = {
    "model": "qwen3.6-27b-uncensored-hauhaucs-balanced",
    "messages": [
        {"role": "system", "content": "Отвечай строго JSON."},
        {"role": "user", "content": "Сформируй JSON {\"result\": \"success\", \"code\": 200}"}
    ],
    "max_tokens": 100,
    "temperature": 0.0
}

req2 = urllib.request.Request(url, data=json.dumps(payload_json).encode('utf-8'), headers={"Content-Type": "application/json"})
resp2 = json.loads(urllib.request.urlopen(req2).read().decode('utf-8'))
content2 = resp2["choices"][0]["message"]["content"]

if "<think>" in content2 and "</think>" in content2:
    content2 = content2.split("</think>")[-1]

match = re.search(r'\{.*\}', content2, re.DOTALL)
raw_json = match.group(0) if match else content2

try:
    parsed = json.loads(raw_json)
    json_ok = "result" in parsed and parsed["result"] == "success"
except Exception:
    json_ok = False

print(f"AST_OK: {ast_ok}")
print(f"JSON_OK: {json_ok}")
print(f"CODE:\n{code_clean[:120]}...")
