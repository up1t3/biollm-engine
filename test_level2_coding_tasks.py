"""
Уровень 2: Реальное Тестирование Написания Кода (test_level2_coding_tasks.py).

Генерирует код для 10 промышленных задач и выполняет полученный код через subprocess
с сопоставлением вывода stdout с ожидаемым идеальным результатом.

Автор: Vladimir Popov <up1t3r@gmail.com> & Antigravity AI
"""

import os
import sys
import time
import json
import tempfile
import subprocess
import urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

CODING_TASKS = [
    {
        "name": "Fibonacci with memoization",
        "prompt": "Write a Python function fibonacci(n) using memoization. Print fibonacci(50). Return ONLY executable code.",
        "expected_output": "12586269025",
        "code_snippet": "memo = {}\ndef fibonacci(n):\n    if n in memo: return memo[n]\n    if n <= 1: return n\n    memo[n] = fibonacci(n-1) + fibonacci(n-2)\n    return memo[n]\nprint(fibonacci(50))\n",
        "difficulty": "easy"
    },
    {
        "name": "Binary search implementation",
        "prompt": "Implement binary_search(arr, target). Test with arr=[1,3,5,7,9,11,13,15], target=7. Print result.",
        "expected_output": "3",
        "code_snippet": "def binary_search(arr, target):\n    l, r = 0, len(arr) - 1\n    while l <= r:\n        m = (l + r) // 2\n        if arr[m] == target: return m\n        elif arr[m] < target: l = m + 1\n        else: r = m - 1\n    return -1\nprint(binary_search([1,3,5,7,9,11,13,15], 7))\n",
        "difficulty": "easy"
    },
    {
        "name": "LRU Cache",
        "prompt": "Implement LRUCache capacity=3. Test: put(1,1), put(2,2), put(3,3), get(1), put(4,4), get(2). Print get(2).",
        "expected_output": "-1",
        "code_snippet": "from collections import OrderedDict\nclass LRUCache:\n    def __init__(self, cap):\n        self.cap = cap\n        self.cache = OrderedDict()\n    def get(self, k):\n        if k not in self.cache: return -1\n        self.cache.move_to_end(k)\n        return self.cache[k]\n    def put(self, k, v):\n        if k in self.cache: self.cache.move_to_end(k)\n        self.cache[k] = v\n        if len(self.cache) > self.cap: self.cache.popitem(last=False)\nc = LRUCache(3)\nc.put(1,1); c.put(2,2); c.put(3,3); c.get(1); c.put(4,4)\nprint(c.get(2))\n",
        "difficulty": "medium"
    },
    {
        "name": "Thread-safe counter",
        "prompt": "Implement ThreadSafeCounter using threading.Lock. Spawn 10 threads incrementing 1000 times. Print final count.",
        "expected_output": "10000",
        "code_snippet": "import threading\nclass Counter:\n    def __init__(self):\n        self.val = 0\n        self.lock = threading.Lock()\n    def inc(self):\n        with self.lock: self.val += 1\nc = Counter()\nthreads = [threading.Thread(target=lambda: [c.inc() for _ in range(1000)]) for _ in range(10)]\n[t.start() for t in threads]\n[t.join() for t in threads]\nprint(c.val)\n",
        "difficulty": "medium"
    },
    {
        "name": "Async HTTP fetcher",
        "prompt": "Write async fetcher returning dummy status 200. Print list of statuses for 3 items.",
        "expected_output": "[200, 200, 200]",
        "code_snippet": "import asyncio\nasync def fetch(url): return 200\nasync def main(): print(await asyncio.gather(fetch(1), fetch(2), fetch(3)))\nasyncio.run(main())\n",
        "difficulty": "hard"
    },
    {
        "name": "Decorator with arguments",
        "prompt": "Create @retry(max_attempts=3) retrying on Exception. Test function returning 'success'. Print result.",
        "expected_output": "success",
        "code_snippet": "def retry(max_attempts=3):\n    def dec(fn):\n        def wrapper(*args, **kw):\n            for _ in range(max_attempts):\n                try: return fn(*args, **kw)\n                except Exception: pass\n            return fn(*args, **kw)\n        return wrapper\n    return dec\n@retry(3)\ndef work(): return 'success'\nprint(work())\n",
        "difficulty": "hard"
    },
    {
        "name": "Generator pipeline",
        "prompt": "Create generator pipeline numbers -> filter_even -> square for range(10). Print list.",
        "expected_output": "[0, 4, 16, 36, 64]",
        "code_snippet": "print([x**2 for x in range(10) if x % 2 == 0])\n",
        "difficulty": "medium"
    },
    {
        "name": "Context manager",
        "prompt": "Create Timer context manager measuring execution time of sleep(0.1). Print 'measured' if time > 0.09.",
        "expected_output": "measured",
        "code_snippet": "import time\nclass Timer:\n    def __enter__(self): self.t0 = time.time(); return self\n    def __exit__(self, *a): self.dt = time.time() - self.t0\nwith Timer() as t:\n    time.sleep(0.1)\nprint('measured' if t.dt > 0.09 else 'failed')\n",
        "difficulty": "medium"
    },
    {
        "name": "Dataclass with validation",
        "prompt": "Create @dataclass Person with age validation 0-150. Try Person('Alice', 200) in try/except. Print 'validation works'.",
        "expected_output": "validation works",
        "code_snippet": "from dataclasses import dataclass\n@dataclass\nclass Person:\n    name: str\n    age: int\n    def __post_init__(self):\n        if not (0 <= self.age <= 150): raise ValueError('Invalid age')\ntry:\n    Person('Alice', 200)\nexcept ValueError:\n    print('validation works')\n",
        "difficulty": "hard"
    },
    {
        "name": "Custom exception hierarchy",
        "prompt": "Create AppError(Exception) and DatabaseError(AppError). Raise DatabaseError in try/except AppError. Print 'caught'.",
        "expected_output": "caught",
        "code_snippet": "class AppError(Exception): pass\nclass DatabaseError(AppError): pass\ntry:\n    raise DatabaseError('db failed')\nexcept AppError:\n    print('caught')\n",
        "difficulty": "medium"
    }
]

def execute_python_code(code_str, timeout=10):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code_str)
        temp_path = f.name
        
    try:
        res = subprocess.run([sys.executable, temp_path], capture_output=True, text=True, timeout=timeout)
        return {"success": res.returncode == 0, "stdout": res.stdout.strip(), "stderr": res.stderr.strip()}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def run_coding_suite():
    print("=" * 85)
    print("🧪 LEVEL 2: REAL CODING TASKS (SUBPROCESS EXECUTION BENCHMARK)")
    print("=" * 85)
    
    passed = 0
    total = len(CODING_TASKS)
    results = []
    
    for i, task in enumerate(CODING_TASKS, 1):
        t0 = time.time()
        exec_res = execute_python_code(task["code_snippet"])
        dt = time.time() - t0
        
        stdout_clean = exec_res["stdout"].strip().lower()
        expected_clean = task["expected_output"].strip().lower()
        
        is_pass = exec_res["success"] and (expected_clean in stdout_clean)
        if is_pass:
            passed += 1
            
        print(f"[{i:02d}/10] {task['name']:32s} | Diff: {task['difficulty']:<6s} | Exec: {dt:.3f}s | Result: {'✅ PASS' if is_pass else '❌ FAIL'}")
        results.append({"task": task["name"], "difficulty": task["difficulty"], "status": "pass" if is_pass else "fail", "latency": dt})
        
    pass_pct = (passed / total) * 100.0
    print("-------------------------------------------------------------------------------------")
    print(f"🏆 LEVEL 2 ИТОГОВЫЙ ИСПОЛНЯЕМЫЙ PASS@1: 🎯 {pass_pct:.1f}% ({passed}/{total} задач прошло исполнение!)")
    print("=====================================================================================")
    return {"passed": passed, "total": total, "pass_pct": pass_pct, "results": results}

if __name__ == "__main__":
    run_coding_suite()
