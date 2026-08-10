"""
Модуль Производственных Шаблонов Промптов BioLLM (biollm_prompt_templates.py).

Устраняет Semantic Mismatch между намерениями пользователя и выходом модели
путем контрактизации роли генератора и маловыборочного форматирования (Few-Shot Context).

Автор: Vladimir Popov <up1t3r@gmail.com> & Antigravity AI
"""

import sys

class BioLLMPromptTemplates:
    @staticmethod
    def format_code_refactor(code_snippet: str, requirements: str = "Использовать aiohttp, добавить type hints и logging") -> str:
        """Шаблон рефакторинга кода"""
        return f"""[SYSTEM ROLE: Senior Python Software Engineer & Performance Architect]
[TASK: Code Refactoring & Async Modernization]

Проведи глубокий рефакторинг следующего Python кода:
```python
{code_snippet}
```

ТРЕБОВАНИЯ К РЕФАКТОРИНГУ:
1. {requirements}.
2. Заменить синхронные блоки/ThreadPoolExecutor на труъ-асинхронные конструкции async/await.
3. Добавить полную статическую типизацию (typing.List, typing.Tuple, typing.Optional).
4. Добавить логирование через модуль logging.
5. Код должен быть на 100% синтаксически корректен и готов к продакшну.
"""

    @staticmethod
    def format_http_server(framework: str = "aiohttp", endpoints: str = "GET /health, POST /api/data") -> str:
        """Шаблон генерации HTTP REST сервера"""
        return f"""[SYSTEM ROLE: Senior Backend Architect & Cloud Systems Engineer]
[TASK: Production REST HTTP Server Generation]

Создай асинхронный HTTP REST сервер на базе {framework}.

ТРЕБОВАНИЯ К СЕРВЕРУ:
1. Поддержка эндпоинтов: {endpoints}.
2. Включить обработку ошибок (HTTP 400 Bad Request, 500 Internal Error) в формате JSON.
3. Включить точку входа запуска через asyncio.run() или web.run_app().
4. Добавить type hints и docstring.
"""

    @staticmethod
    def format_bug_fix(code_snippet: str, error_description: str) -> str:
        """Шаблон диагностики и исправления ошибок"""
        return f"""[SYSTEM ROLE: Principal Systems Debugger]
[TASK: Bug Fix & Diagnosis]

Найди и исправь ошибку в коде:
```python
{code_snippet}
```
ОПИСАНИЕ ОШИБКИ: {error_description}

ТРЕБОВАНИЯ:
1. Предоставить исправленный код.
2. Кратко объяснить первопричину багов (Root Cause Analysis).
"""

    @staticmethod
    def format_algorithm(algo_name: str, requirements: str = "Type hints, docstrings, example usage") -> str:
        """Шаблон генерации алгоритмов и структур данных"""
        return f"""[SYSTEM ROLE: Algorithm & Data Structures Specialist]
[TASK: Algorithm Implementation]

Реализуй алгоритм {algo_name} на Python.

ТРЕБОВАНИЯ:
1. {requirements}.
2. Оптимальная алгоритмическая сложность по времени O(N) и памяти O(1).
3. Добавить пример использования с assert проверками.
"""

if __name__ == "__main__":
    print("🧪 Тестирование BioLLMPromptTemplates...")
    sample_refactor = BioLLMPromptTemplates.format_code_refactor(
        "def fetch(url): return urllib.request.urlopen(url).read()",
        "Заменить urllib на aiohttp"
    )
    print(sample_refactor[:200] + "...")
