"""
Скрипт детального замера скорости генерации токенов в секунду (tok/s speed benchmark).
Измеряет tok/s mean, p50, p95, Time-To-First-Token (TTFT) и задержку на токен (ms/tok).
"""

import os
import sys
import time
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from biollm_model import BioAutoModelForCausalLM

# Настройка UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

MODEL_PATH = r"C:\Users\Up1t3\.gemini\antigravity\scratch\biollm\converted_models\qwen_bio.biollm"

def benchmark_token_speed():
    print("=" * 85)
    print("⚡ ЗАМЕР СКОРОСТИ ГЕНЕРАЦИИ ТОКЕНОВ В СЕКУНДУ (TOKENS PER SECOND BENCHMARK)")
    print("=" * 85)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Модель не найдена по пути: {MODEL_PATH}")
        sys.exit(1)

    model = BioAutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.eval()

    prompt_tokens = torch.tensor([[101, 2054, 2003, 1037, 3899, 102]], dtype=torch.long)
    token_lengths = [64, 128, 256]

    print("\n1. Проведение серии замеров длины вывода (64, 128, 256 токенов):")
    print("-" * 85)
    print(f"{'Длина вывода':<20} | {'Время (сек)':<15} | {'Скорость (tok/s)':<18} | {'ms / токен':<15} | {'TTFT (ms)'}")
    print("-" * 85)

    all_speeds = []

    for max_new in token_lengths:
        # Измерение Time-To-First-Token (TTFT)
        start_ttft = time.time()
        _ = model.generate(prompt_tokens, max_new_tokens=1, enable_telemetry=False)
        ttft_ms = (time.time() - start_ttft) * 1000

        # Измерение полного цикла генерации
        start_gen = time.time()
        res = model.generate(prompt_tokens, max_new_tokens=max_new, enable_telemetry=True)
        elapsed = time.time() - start_gen

        generated_tokens = len(res["output_ids"][0])
        tok_s = res["tokens_per_second"]
        ms_per_tok = (elapsed / max(generated_tokens, 1)) * 1000

        all_speeds.append(tok_s)

        print(f"{generated_tokens:>5} токенов         | {elapsed:>12.3f} s | {tok_s:>15.2f} tok/s | {ms_per_tok:>12.2f} ms | {ttft_ms:>10.2f} ms")

    all_speeds.sort()
    p50_speed = all_speeds[int(len(all_speeds) * 0.5)]
    p95_speed = all_speeds[int(len(all_speeds) * 0.95)]
    mean_speed = sum(all_speeds) / len(all_speeds)

    print("-" * 85)
    print("\n" + "=" * 85)
    print("📊 ИТОГОВЫЙ ПАСПОРТ СКОРОСТИ BioLLM ENGINE v3.5:")
    print("=" * 85)
    print(f"Средняя скорость (tok/s mean):   {mean_speed:.2f} tok/s")
    print(f"Квантиль скорости p50:           {p50_speed:.2f} tok/s")
    print(f"Квантиль скорости p95:           {p95_speed:.2f} tok/s")
    print(f"Средняя задержка на 1 токен:     {(1000/mean_speed):.2f} ms / token")
    print(f"Time-To-First-Token (TTFT):       {ttft_ms:.2f} ms")
    print("=" * 85)
    print("✅ ЗАМЕР СКОРОСТИ УСПЕШНО ЗАВЕРШЕН.")

if __name__ == "__main__":
    benchmark_token_speed()
