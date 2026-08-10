"""
База Знаний Асинхронных Паттернов Python (biollm_async_knowledge.py).

Содержит проверенные экспертные шаблоны и правила для генератора:
- Управление конкурентностью через asyncio.Semaphore
- Повторное использование aiohttp.ClientSession
- Безопасная обработка исключений asyncio.gather(return_exceptions=True)
- Избежание blocking I/O (urllib, requests, time.sleep)

Автор: Vladimir Popov <up1t3r@gmail.com> & Antigravity AI
"""

import sys

ASYNC_BEST_PRACTICES = {
    "session_reuse": "Всегда используйте единую сессию aiohttp.ClientSession(connector=TCPConnector(limit=N)) вместо создания новой сессии на каждый запрос.",
    "concurrency_control": "Для предотвращения OOM или сетевой перегрузки вызовов используйте asyncio.Semaphore(max_concurrent) для пулирования задач.",
    "exception_handling": "При вызове asyncio.gather() передавайте return_exceptions=True, чтобы сбой одного запроса не отменял всю параллельную группу.",
    "non_blocking_sleep": "Категорически заменять time.sleep() на await asyncio.sleep() для сохранения производительности основного event loop."
}

def get_async_knowledge_prompt() -> str:
    """Форматирует базу знаний асинхронности в системный контекст промпта"""
    rules = "\n".join([f"- {k.upper()}: {v}" for k, v in ASYNC_BEST_PRACTICES.items()])
    return f"""[ASYNC KNOWLEDGE BASE & BEST PRACTICES]
{rules}
"""

if __name__ == "__main__":
    print(get_async_knowledge_prompt())
