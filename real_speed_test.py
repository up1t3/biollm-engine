"""
Реальный Скрипт Замера Скорости Инференса BioLLM REST API (real_speed_test.py).
Оценивает честную пропускную способность вычисления токенов через HTTP API.
"""

import time
import json
import sys
import urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def query_model(prompt, max_tokens=200):
    url = "http://localhost:8000/v1/chat/completions"
    data = {
        "model": "biollm-ornith-35b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    
    t_start = time.perf_counter()
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read())
    t_end = time.perf_counter()
    
    elapsed = t_end - t_start
    content = result["choices"][0]["message"]["content"]
    
    tokens_approx = len(content.split()) + content.count(",") + content.count(".")
    
    return {
        "time_seconds": elapsed,
        "tokens_approx": tokens_approx,
        "tokens_per_sec": tokens_approx / elapsed if elapsed > 0 else 0,
        "content_preview": content[:200]
    }

print("=" * 60)
print("TEST 1: Short response")
print("=" * 60)
r1 = query_model("Что такое 2+2? Ответь одним словом.", max_tokens=10)
print(f"Time:   {r1['time_seconds']:.3f}s")
print(f"Tokens: {r1['tokens_approx']}")
print(f"Speed:  {r1['tokens_per_sec']:.1f} tok/s")
print(f"Output: {r1['content_preview']}")

print("\n" + "=" * 60)
print("TEST 2: Medium code generation")
print("=" * 60)
r2 = query_model(
    "Напиши функцию Python для проверки палиндрома. Без объяснений, только код.",
    max_tokens=150
)
print(f"Time:   {r2['time_seconds']:.3f}s")
print(f"Tokens: {r2['tokens_approx']}")
print(f"Speed:  {r2['tokens_per_sec']:.1f} tok/s")
print(f"Output:\n{r2['content_preview']}")

print("\n" + "=" * 60)
print("TEST 3: Long generation")
print("=" * 60)
r3 = query_model(
    "Напиши подробный туториал по async/await в Python на 500 слов.",
    max_tokens=800
)
print(f"Time:   {r3['time_seconds']:.3f}s")
print(f"Tokens: {r3['tokens_approx']}")
print(f"Speed:  {r3['tokens_per_sec']:.1f} tok/s")

print("\n" + "=" * 60)
print("ИТОГОВАЯ СРЕДНЯЯ СКОРОСТЬ:")
print("=" * 60)
avg_speed = (r1['tokens_per_sec'] + r2['tokens_per_sec'] + r3['tokens_per_sec']) / 3
print(f"Средняя скорость: {avg_speed:.1f} tok/s")
print(f"Честный предел RTX 3090 для 35B: ~95.5 tok/s")
print(f"Достигнуто: {avg_speed/95.5*100:.1f}% от теоретического максимума")
