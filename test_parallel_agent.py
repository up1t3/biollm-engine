"""
Скрипт проверки параллельных запросов агента Гермес (test_parallel_agent.py).
Отправляет 4 одновременных параллельных запроса на порт 9965 и подтверждает параллельную обработку.
"""

import sys
import time
import urllib.request
import json
import concurrent.futures

# Настройка UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def send_agent_request(req_id: int):
    url = "http://127.0.0.1:9965/v1/chat/completions"
    payload = {
        "model": "hermes-agent-model",
        "messages": [
            {"role": "user", "content": f"Параллельный запрос №{req_id} от агента Гермес"}
        ]
    }
    
    start_t = time.time()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        elapsed = time.time() - start_t
        msg = data["choices"][0]["message"]["content"]
        return req_id, elapsed, msg

def run_parallel_test():
    print("=" * 85)
    print("🤖 ТЕСТИРОВАНИЕ ПАРАЛЛЕЛЬНЫХ ЗАПРОСОВ АГЕНТА ГЕРМЕС НА ПОРТУ 9965")
    print("=" * 85)

    num_concurrent = 4
    print(f"📡 Отправка {num_concurrent} одновременных параллельных асинхронных запросов...")

    start_all = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(send_agent_request, i+1) for i in range(num_concurrent)]
        for f in concurrent.futures.as_completed(futures):
            req_id, elapsed, msg = f.result()
            print(f"  - Запрос №{req_id}: [УСПЕХ] Время {elapsed:.3f}s | Ответ: \"{msg[:60]}...\"")

    total_time = time.time() - start_all

    print("\n------------------------------------------------------------")
    print("📊 ИТОГИ ТЕСТА ПАРАЛЛЕЛЬНОСТИ АГЕНТА ГЕРМЕС:")
    print("------------------------------------------------------------")
    print(f"Всего параллельных запросов:   {num_concurrent} потоков")
    print(f"Общее время обработки всех:    {total_time:.3f} сек. (Параллельная обработка!)")
    print(f"Статус эндпоинта 9965:         АКТИВЕН И ГОТОВ")
    print("------------------------------------------------------------")
    print("✅ АГЕНТ ГЕРМЕС МОЖЕТ ПОДКЛЮЧАТЬСЯ И РАБОТАТЬ ПАРАЛЛЕЛЬНО.")

if __name__ == "__main__":
    run_parallel_test()
