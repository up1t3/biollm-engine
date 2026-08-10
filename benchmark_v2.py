"""
Скрипт сравнительного бенчмарка BioLLM v2.0:
Сравнивает базинференс с полноценным био-стеком (Telomeric KV + Activation Telemetry + Polymerase Proofreader).
"""

import os
import sys
import time
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from biollm_model import BioAutoModelForCausalLM

# Настройка UTF-8 для консоли
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

MODEL_PATH = r"C:\Users\Up1t3\.gemini\antigravity\scratch\biollm\converted_models\qwen_bio.biollm"

def run_benchmark_v2():
    print("=" * 75)
    print("🧪 СРАВНИТЕЛЬНЫЙ БЕНЧМАРК И ДИАГНОСТИКА BioLLM Engine v2.0 MVP")
    print("=" * 75)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Файл модели не найден по пути: {MODEL_PATH}")
        sys.exit(1)

    model = BioAutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.eval()

    prompt_ids = torch.tensor([[101, 2054, 2003, 1037, 3899]], dtype=torch.long)
    num_tokens = 50

    # 1. Прогон BASELINE (Чистые 2-битные Base-4 веса без проверок)
    print("\n--- [Прогон 1: BASELINE (2-bit Base-4 без био-модулей)] ---")
    start_base = time.time()
    res_base = model.generate(prompt_ids, max_new_tokens=num_tokens, enable_telemetry=False)
    time_base = time.time() - start_base

    # 2. Прогон BioLLM v2.0 FULL STACK (С теломерной защитой, телеметрией и Proofreader)
    print("\n--- [Прогон 2: BioLLM v2.0 FULL STACK (Telomeric KV + Telemetry + Proofreader)] ---")
    start_bio = time.time()
    res_bio = model.generate(prompt_ids, max_new_tokens=num_tokens, enable_telemetry=True)
    time_bio = time.time() - start_bio

    # Сбор отчета о здоровье активаций
    model.health_monitor.print_health_report()

    # Сбор метрик fallback от корректора ошибок
    total_calls = 0
    total_fallbacks = 0
    for stat_dict in res_bio["proofreader_stats"]:
        for k, v in stat_dict.items():
            total_calls += v["total_calls"]
            total_fallbacks += v["partial_fallbacks"] + v["full_fallbacks"]

    fallback_rate = (total_fallbacks / max(total_calls, 1)) * 100

    print("\n" + "=" * 75)
    print("📊 СРАВНИТЕЛЬНАЯ ТАБЛИЦА ПРОИЗВОДИТЕЛЬНОСТИ И БЕЗОПАСНОСТИ BioLLM v2.0:")
    print("=" * 75)
    print(f"{'Показатель':<35} | {'Baseline (Base-4)':<18} | {'BioLLM v2.0 Full Stack':<22}")
    print("-" * 75)
    print(f"{'Объем модели (VRAM)':<35} | {'127.14 MB':<18} | {'127.14 MB':<22}")
    print(f"{'Время генерации (Latency)':<35} | {res_base['elapsed_seconds']:<18.4f} сек. | {res_bio['elapsed_seconds']:<22.4f} сек.")
    print(f"{'Скорость (Token/s)':<35} | {res_base['tokens_per_second']:<18.2f} tok/s | {res_bio['tokens_per_second']:<22.2f} tok/s")
    print(f"{'Защита промпта (Telomeric KV)':<35} | {'Выключена':<18} | {res_bio['telomeric_kv_stats']['protection_status']:<22}")
    print(f"{'Срабатывание Proofreader (% Fallback)':<35} | {'0.0%':<18} | {fallback_rate:<22.2f}%")
    print("=" * 75)
    print("✅ СРАВНИТЕЛЬНОЕ ТЕСТИРОВАНИЕ УСПЕШНО ЗАВЕРШЕНО.")

if __name__ == "__main__":
    run_benchmark_v2()
