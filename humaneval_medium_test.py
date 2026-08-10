"""
Бенчмарк Небытовых Задач Алгоритмов Повышенной Сложности (humaneval_medium_test.py).
Проверяет функциональную проходимость (pass@1) 3 классических сложных задач LeetCode Medium / HumanEval:
1. Two Sum (Поиск индексов в массиве)
2. Valid Parentheses (Стек скобочных структур)
3. Merge Overlapping Intervals (Слияние пересекающихся интервалов)
"""

import ast
import subprocess
import sys
import tempfile
import os
import json
import urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def generate_code(task_description):
    url = "http://localhost:8085/v1/chat/completions"
    data = {
        "model": "biollm-ornith-35b-stream",
        "messages": [
            {"role": "system", "content": "You are a Python expert. Return ONLY valid python code."},
            {"role": "user", "content": task_description}
        ],
        "max_tokens": 500,
        "temperature": 0.0
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read())
    
    content = result["choices"][0]["message"]["content"]
    if "```python" in content:
        content = content.split("```python")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
        
    return content.strip()

def execute_task(task_name, description, runner_code):
    print(f"\n======================================================================")
    print(f"🔬 MEDIUM BENCHMARK TASK: [{task_name}]")
    print(f"======================================================================")
    
    solution_code = generate_code(description)
    print(f"Generated Code:\n{solution_code[:250]}...\n")
    
    full_code = f"{solution_code}\n\n{runner_code}"
    
    try:
        ast.parse(full_code)
        ast_valid = True
    except SyntaxError as e:
        print(f"❌ AST Syntax Error: {e}")
        return False
        
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(full_code)
        f.flush()
        temp_file = f.name
        
    try:
        res = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=10
        )
        success = (res.returncode == 0 and "TESTS_PASSED" in res.stdout)
        print(f"AST Valid: {ast_valid}")
        print(f"Stdout Output:\n{res.stdout.strip()}")
        if not success and res.stderr:
            print(f"Stderr Output:\n{res.stderr.strip()}")
        print(f"RESULT: {'✅ PASS' if success else '❌ FAIL'}")
        return success
    except Exception as e:
        print(f"❌ Execution Exception: {e}")
        return False
    finally:
        os.unlink(temp_file)

# 1. Task Two Sum
runner_1 = """
res1 = two_sum([2, 7, 11, 15], 9)
res2 = two_sum([3, 2, 4], 6)
assert res1 == [0, 1], f"Expected [0, 1], got {res1}"
assert res2 == [1, 2], f"Expected [1, 2], got {res2}"
print("TESTS_PASSED")
"""

# 2. Task Valid Parentheses
runner_2 = """
assert is_valid_parentheses("()") == True
assert is_valid_parentheses("()[]{}") == True
assert is_valid_parentheses("(]") == False
assert is_valid_parentheses("([)]") == False
print("TESTS_PASSED")
"""

# 3. Task Merge Intervals
runner_3 = """
res1 = merge_intervals([[1,3],[2,6],[8,10],[15,18]])
res2 = merge_intervals([[1,4],[4,5]])
assert res1 == [[1,6],[8,10],[15,18]], f"Expected [[1,6],[8,10],[15,18]], got {res1}"
assert res2 == [[1,5]], f"Expected [[1,5]], got {res2}"
print("TESTS_PASSED")
"""

def run_humaneval_medium():
    # Запускаем локальный серверок в фоновом процессе
    from biollm_fastapi_bridge import run_streaming_server
    import threading
    t = threading.Thread(target=run_streaming_server, daemon=True)
    t.start()
    time.sleep(1)
    
    results = []
    results.append(execute_task("Two Sum (Hash Map Lookup)", "Write function two_sum(nums, target)", runner_1))
    results.append(execute_task("Valid Parentheses (Stack Parsing)", "Write function is_valid_parentheses(s)", runner_2))
    results.append(execute_task("Merge Overlapping Intervals (Sorting & Intervals)", "Write function merge_intervals(intervals)", runner_3))
    
    print("\n" + "=" * 70)
    print("📊 ИТОГ БЕНЧМАРКА HUMAN-EVAL MEDIUM / LEETCODE:")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"  • Итоговый Pass@1 (Medium Tasks): 🎯 {passed}/{total} ({passed/total*100:.1f}%)")
    print("======================================================================")

if __name__ == "__main__":
    import time
    run_humaneval_medium()
