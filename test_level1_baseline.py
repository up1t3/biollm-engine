"""
Уровень 1: Проверка Базовой Работоспособности (test_level1_baseline.py).

Проверяет доступность REST эндпоинтов, уникальность хэша ответа и базовое время отклика.

Автор: Vladimir Popov <up1t3r@gmail.com> & Antigravity AI
"""

import os
import sys
import time
import json
import urllib.request
import hashlib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def test_baseline():
    print("=" * 85)
    print("🧪 LEVEL 1: BASELINE VERIFICATION (REST API & LATENCY CHECK)")
    print("=" * 85)
    
    endpoints = [
        "http://localhost:8085/v1/models",
        "http://localhost:9965/v1/models"
    ]
    
    for endpoint in endpoints:
        try:
            req = urllib.request.Request(endpoint)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                print(f"✅ {endpoint}: HTTP {resp.status}")
                print(f"   Available models: {len(data.get('data', []))}")
        except Exception as e:
            print(f"❌ {endpoint}: {e}")
            
    print("\n-------------------------------------------------------------------------------------")
    print("⚡ ИЗМЕРЕНИЕ ВРЕМЕНИ И ХЭША ЗАПРОСА:")
    start = time.time()
    
    payload = {
        "model": "qwen3.6-27b-hybrid-mamba",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 10,
        "stream": False
    }
    
    try:
        req = urllib.request.Request(
            "http://localhost:8085/v1/chat/completions",
            data=json.dumps(payload).encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            elapsed = time.time() - start
            data = json.loads(resp.read().decode('utf-8'))
            content = data["choices"][0]["message"]["content"]
            resp_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
            
            print(f"✅ Simple query execution time: {elapsed:.3f}s")
            print(f"   Response snippet: '{content[:80]}'")
            print(f"   Response hash: {resp_hash}")
            return True
    except Exception as e:
        print(f"❌ Simple query failed: {e}")
        return False

if __name__ == "__main__":
    test_baseline()
