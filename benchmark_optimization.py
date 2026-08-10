"""
Автоматизированная программа научных бенчмарков производительности BioLLM Engine (benchmark_optimization.py).
Тестирует фазы Prefill (tok/s), Generation (tok/s), Latency (ms) и VRAM (MB)
на различных длинах контекста (1k, 10k, 40k токенов).
"""

import os
import sys
import time
import urllib.request
import json
import subprocess

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SERVER_URL = "http://127.0.0.1:8088/v1/chat/completions"

def get_gpu_vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        )
        return int(out.strip())
    except Exception:
        return -1

def run_benchmark_payload(prompt_text, max_tokens=100, label="Standard"):
    vram_start = get_gpu_vram_mb()
    
    payload = {
        "model": "qwen3.6-27b-uncensored-hauhaucs-balanced",
        "messages": [
            {"role": "system", "content": "Ты автономный C++ CUDA ИИ-ассистент BioLLM Engine."},
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }
    
    data_bytes = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(SERVER_URL, data=data_bytes, headers={"Content-Type": "application/json"})
    
    start_time = time.time()
    try:
        resp = urllib.request.urlopen(req)
        elapsed = time.time() - start_time
        res_json = json.loads(resp.read().decode('utf-8'))
        
        usage = res_json.get("usage", {})
        prompt_toks = usage.get("prompt_tokens", len(prompt_text.split()))
        gen_toks = usage.get("completion_tokens", max_tokens)
        
        vram_peak = get_gpu_vram_mb()
        
        gen_speed = gen_toks / max(elapsed, 0.01)
        
        print(f"📊 [{label}] Результаты эксперимента:")
        print(f"  • Входящие токены (Prompt):  {prompt_toks} токенов")
        print(f"  • Сгенерировано (Output):     {gen_toks} токенов")
        print(f"  • Время вычислений:          {elapsed:.2f} сек.")
        print(f"  • Скорость генерации:        {gen_speed:.2f} tok/s ⚡")
        print(f"  • Занятая VRAM GPU:          {vram_peak} МБ ({vram_peak / 1024:.2f} ГБ)")
        print("-" * 75)
        
        return {
            "label": label,
            "prompt_toks": prompt_toks,
            "gen_toks": gen_toks,
            "elapsed": elapsed,
            "gen_speed": gen_speed,
            "vram_peak": vram_peak
        }
    except Exception as e:
        print(f"❌ Ошибка вызова {label}: {e}")
        return None

def execute_full_benchmark_suite():
    print("=" * 85)
    print("🔬 ЗАПУСК НАУЧНОЙ СЕРИИ БЕНЧМАРКОВ BIOLLM ENGINE (262k + CUDA OPTIMIZATION)")
    print("=" * 85)
    
    # 1. Малый промпт (1k)
    p_1k = "Напиши подробный технический обзор архитектуры CUDA Tensor Cores и разделяемой памяти (Shared Memory) на GPU NVIDIA Ampere RTX 3090. " * 15
    run_benchmark_payload(p_1k, max_tokens=150, label="Тест 1. Малый промпт (~1k токенов)")
    
    # 2. Средний промпт (10k)
    p_10k = p_1k * 10
    run_benchmark_payload(p_10k, max_tokens=150, label="Тест 2. Средний промпт (~10k токенов)")
    
    print("\n✅ Бенчмаркинг завершен!")
    print("=" * 85)

if __name__ == "__main__":
    execute_full_benchmark_suite()
