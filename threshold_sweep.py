"""
Модуль автоматического подбора и калибровки порогов (Proofreader Threshold Sweep).
Проводит свип порогов аномальности tau in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
для поиска идеальной "elbow-точки" (где fallback_rate_total < 5%, а качество максимально).
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

THRESHOLDS_TO_TEST = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

def run_threshold_sweep():
    print("=" * 80)
    print("🔬 АВТОМАТИЧЕСКАЯ КАЛИБРОВКА ПОРОГОВ (PROOFREADER THRESHOLD SWEEP)")
    print("=" * 80)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Файл модели не найден по пути: {MODEL_PATH}")
        sys.exit(1)

    model = BioAutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.eval()

    prompt_tokens = torch.tensor([[101, 2054, 2003, 1037, 3899]], dtype=torch.long)
    num_tokens = 40

    sweep_results = []
    best_threshold = None
    min_dist_to_target = 999.0

    print(f"{'Threshold (tau)':<15} | {'Fallback Rate (%)':<20} | {'Latency (sec)':<15} | {'Speed (tok/s)':<15}")
    print("-" * 75)

    for tau in THRESHOLDS_TO_TEST:
        # Устанавливаем новый порог для всех слоев proofreader
        for layer in model.layers:
            layer.mlp_gate.spike_threshold = tau
            layer.mlp_down.spike_threshold = tau
            # Сброс статистики
            layer.mlp_gate.total_calls = 0
            layer.mlp_gate.partial_fallbacks = 0
            layer.mlp_gate.full_fallbacks = 0

        start_t = time.time()
        res = model.generate(prompt_tokens, max_new_tokens=num_tokens, enable_telemetry=True)
        elapsed = time.time() - start_t

        # Сбор показателей fallback
        total_calls = 0
        total_fallbacks = 0
        for stat_dict in res["proofreader_stats"]:
            for _, v in stat_dict.items():
                total_calls += v["total_calls"]
                total_fallbacks += v["partial_fallbacks"] + v["full_fallbacks"]

        fb_rate = (total_fallbacks / max(total_calls, 1)) * 100
        tok_s = num_tokens / elapsed

        sweep_results.append({
            "threshold": tau,
            "fallback_rate": fb_rate,
            "latency": elapsed,
            "tokens_per_sec": tok_s
        })

        print(f"{tau:<15.1f} | {fb_rate:<20.2f}% | {elapsed:<15.4f} | {tok_s:<15.2f}")

        # Поиск оптимальной точки (целевой fallback_rate около 5.0%)
        dist = abs(fb_rate - 5.0)
        if dist < min_dist_to_target:
            min_dist_to_target = dist
            best_threshold = tau

    print("-" * 75)
    print(f"🎯 ИДОАЛЬНАЯ ТОЧКА ИЗГИБА (ELBOW POINT): Threshold (tau) = {best_threshold:.1f}")
    print(f"При этом пороге fallback_rate составил около {sweep_results[THRESHOLDS_TO_TEST.index(best_threshold)]['fallback_rate']:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    run_threshold_sweep()
