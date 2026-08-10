"""
Скрипт прямого диалога и проверки ответов модели Qwen3.6-27B.
Отправляет текстовый запрос и замеряет скорость, токены и качество ответа.
"""

import sys
import time
import urllib.request
import json

# Настройка UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def test_live_chat():
    print("=" * 85)
    print("💬 ИНИЦИАЛИЗАЦИЯ ЖИВОГО ДИАЛОГА С МОДЕЛЬЮ QWEN3.6-27B (BIOLLM STACK)")
    print("=" * 85)

    prompt = "Привет! Расскажи кратко, как ты себя чувствуешь и готова ли ты к работе с длинными контекстами?"
    print(f"👤 Посетитель: \"{prompt}\"\n")

    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 128
    }

    # Попытка подключиться к эндпоинтам (LM Studio на 1234 или BioLLM на 6694 / 56313)
    endpoints = [
        "http://127.0.0.1:1234/v1/chat/completions",
        "http://127.0.0.1:6694/v1/chat/completions",
        "http://127.0.0.1:56313/v1/chat/completions"
    ]

    response_data = None
    used_endpoint = None

    for ep in endpoints:
        try:
            req = urllib.request.Request(
                ep,
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            start_time = time.time()
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    raw = response.read().decode('utf-8')
                    response_data = json.loads(raw)
                    elapsed = time.time() - start_time
                    used_endpoint = ep
                    break
        except Exception:
            continue

    if response_data and "choices" in response_data:
        msg = response_data["choices"][0]["message"]["content"]
        tokens = response_data.get("usage", {}).get("completion_tokens", 30)
        tok_s = tokens / max(elapsed, 0.1)

        print("------------------------------------------------------------")
        print("🤖 Ответ модели Qwen3.6-27B:")
        print("------------------------------------------------------------")
        print(f"{msg}\n")
        print("------------------------------------------------------------")
        print("📊 СТАТИСТИКА ГЕНЕРАЦИИ:")
        print(f"Эндпоинт:                {used_endpoint}")
        print(f"Время ответа:            {elapsed:.2f} сек.")
        print(f"Сгенерировано токенов:   {tokens}")
        print(f"Скорость генерации:      {tok_s:.2f} tok/s")
        print("------------------------------------------------------------")
    else:
        print("⚠️ Не удалось связаться с активным эндпоинтом. Запускаем локальный прямой генератор BioLLM...")
        from interactive_client import test_biollm_interactive
        test_biollm_interactive(prompt)

if __name__ == "__main__":
    test_live_chat()
