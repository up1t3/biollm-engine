"""
Тестовый Скрипт Проверки Запроса к Локальному REST API Серверу VS Code (http://localhost:8000/v1).

Имитирует точно такой же запрос, который делает VS Code (Continue / Cline)
при редактировании и генерации асинхронного Python кода на вашей видеокарте.

Автор: Vladimir Popov <up1t3r@gmail.com> & Antigravity AI
"""

import urllib.request
import json
import time

def test_vscode_rest_request():
    print("=" * 85)
    print("🚀 ИСПЫТАНИЕ ЗАПРОСА VS CODE К ЛОКАЛЬНОМУ СЕРВЕРУ HTTP://LOCALHOST:8000/v1")
    print("=" * 85)
    
    url = "http://localhost:8000/v1/chat/completions"
    payload = {
        "model": "gemma4-local",
        "messages": [
            {"role": "user", "content": "Напиши чистую асинхронную функцию на Python для параллельной загрузки файлов с использованием aiohttp и asyncio.gather."}
        ]
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            t_elapsed = time.time() - t0
            
            reply_text = data['choices'][0]['message']['content']
            
            print(f"✅ УСПЕШНЫЙ ОТКЛИК ЗА {t_elapsed:.2f} СЕКУНД!")
            print("------------------------------------------------------------")
            print("💻 ОТВЕТ МОДЕЛИ ДЛЯ VS CODE:")
            print("------------------------------------------------------------")
            print(reply_text)
            print("=================================================================")
    except Exception as e:
        print(f"⚠️ Ошибка запроса к REST серверу: {e}")

if __name__ == "__main__":
    test_vscode_rest_request()
