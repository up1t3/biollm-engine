"""
Реальный Тест Выполнения Кода (Level 3: test_code_execution.py).
Проверяет функциональную проходимость (pass@1) 5 алгоритмических задач через Python Subprocess execution.
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
    url = "http://localhost:8000/v1/chat/completions"
    data = {
        "model": "biollm-ornith-35b",
        "messages": [
            {"role": "system", "content": "You are a Python expert. Return ONLY the python code, no explanations."},
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

def execute_code(code, test_input=None):
    try:
        ast.parse(code)
        ast_valid = True
    except SyntaxError as e:
        return {"ast_valid": False, "error": f"SyntaxError: {e}"}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.flush()
        temp_file = f.name
    
    try:
        result = subprocess.run(
            [sys.executable, temp_file],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=10
        )
        return {
            "ast_valid": ast_valid,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {"ast_valid": ast_valid, "error": "Timeout"}
    finally:
        os.unlink(temp_file)

tasks = [
    {
        "name": "Fibonacci",
        "description": "Write function fibonacci(n) that returns nth Fibonacci number. Print fibonacci(10) at the end.",
        "expected_output": "55"
    },
    {
        "name": "Prime check",
        "description": "Write function is_prime(n). Print result of is_prime(17) and is_prime(18) as True/False on separate lines.",
        "expected_output": "True\nFalse"
    },
    {
        "name": "Reverse string",
        "description": "Write function reverse_string(s). Print reverse_string('hello world').",
        "expected_output": "dlrow olleh"
    },
    {
        "name": "Sum of list",
        "description": "Write function sum_list(lst) without using sum(). Print sum_list([1,2,3,4,5]).",
        "expected_output": "15"
    },
    {
        "name": "Factorial",
        "description": "Write function factorial(n) iteratively. Print factorial(6).",
        "expected_output": "720"
    }
]

print("=" * 70)
print("REAL CODE EXECUTION TEST")
print("=" * 70)

results = []
for task in tasks:
    print(f"\n[{task['name']}]")
    print("-" * 70)
    
    code = generate_code(task["description"])
    print(f"Generated code:\n{code[:300]}...")
    
    exec_result = execute_code(code)
    print(f"\nAST valid: {exec_result.get('ast_valid', False)}")
    
    if exec_result.get('success'):
        stdout = exec_result['stdout'].strip()
        expected = task['expected_output'].strip()
        correct = stdout == expected
        print(f"Output:   {stdout}")
        print(f"Expected: {expected}")
        print(f"CORRECT:  {correct}")
        results.append(correct)
    else:
        print(f"FAILED: {exec_result.get('stderr', exec_result.get('error'))}")
        results.append(False)

print("\n" + "=" * 70)
print("ИТОГ:")
print("=" * 70)
passed = sum(results)
total = len(results)
print(f"Pass@1: {passed}/{total} ({passed/total*100:.1f}%)")
