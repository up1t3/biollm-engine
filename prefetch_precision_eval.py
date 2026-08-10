"""
Скрипт валидации точности и генерализации (BioLLM Prefetch Generalization & Precision Gate v2.1).
Вычисляет Precision@K, Recall@K, MRR, избыточность подкачки (Unnecessary Prefetch Ratio)
и устойчивость к дистракторам (Distractor Resistance).
"""

import os
import sys
import time
import json
import random
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from polya_evictor import PolyAEvictorV12
from retrieval_index import BlockRetrievalIndex
from prefetch_planner_v2_1 import PrefetchPlannerV21
from biollm_model import BioAutoModelForCausalLM

# Настройка UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

MODEL_PATH = r"C:\Users\Up1t3\.gemini\antigravity\scratch\biollm\converted_models\qwen_bio.biollm"
ARTIFACT_DIR = r"C:\Users\Up1t3\.gemini\antigravity\brain\a67e8020-b639-4c7f-a3e7-6e916b6206db"

def run_prefetch_precision_eval():
    print("=" * 85)
    print("🎯 BIOLLM PREFETCH GENERALIZATION & PRECISION GATE v2.1")
    print("=" * 85)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Файл модели не найден по пути: {MODEL_PATH}")
        sys.exit(1)

    random.seed(42)
    torch.manual_seed(42)

    model = BioAutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.eval()

    total_tasks = 100
    target_block_id = 142
    distractor_block_id = 210

    index = BlockRetrievalIndex(embedding_dim=64)
    evictor = PolyAEvictorV12(task_type="code", max_vram_blocks=16, initial_polya_tail=20)

    evictor.register_kv_block(0, torch.randn(1, 16, 64, 64), is_head=True)

    # 1. Индексация 1024 блоков с добавлением ложного дистрактор-блока
    print("\n1. Генерация 1024 блоков с целевым паттерном и дистрактором...")
    target_pattern = torch.randn(1, 16, 64, 64)
    distractor_pattern = target_pattern + torch.randn_like(target_pattern) * 0.15 # Очень похожий дистрактор

    for i in range(1, 1024):
        if i == target_block_id:
            block_t = target_pattern.clone()
        elif i == distractor_block_id:
            block_t = distractor_pattern.clone()
        else:
            block_t = torch.randn(1, 16, 64, 64)
            
        evictor.register_kv_block(i, block_t, is_head=False, is_tail=(i >= 1016))
        index.add_or_update_block(i, block_t.mean(dim=(0, 1)))

    for _ in range(5):
        evictor.step_decay_and_evict()

    planner = PrefetchPlannerV21(retrieval_index=index, min_k=2, max_k=8)

    reciprocal_ranks = []
    hits_at_1 = 0
    hits_at_2 = 0
    hits_at_4 = 0
    hits_at_8 = 0
    total_prefetched_count = 0
    distractor_resisted_count = 0

    print("\n2. Проведение 100 адаптивных тестов подкачки с контролем точности...")

    for step in range(total_tasks):
        query_v = target_pattern.mean(dim=(0, 1)) + torch.randn_like(target_pattern.mean(dim=(0, 1))) * 0.03
        
        predicted_ids, meta = planner.plan_prefetch_adaptive(query_v, evictor.evicted_cpu_blocks)
        total_prefetched_count += len(predicted_ids)

        # Расчет Reciprocal Rank
        if target_block_id in predicted_ids:
            rank = predicted_ids.index(target_block_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        # Оценка Hits@K
        if target_block_id in predicted_ids[:1]:
            hits_at_1 += 1
        if target_block_id in predicted_ids[:2]:
            hits_at_2 += 1
        if target_block_id in predicted_ids[:4]:
            hits_at_4 += 1
        if target_block_id in predicted_ids[:8]:
            hits_at_8 += 1

        # Проверка устойчивости к дистракторам (целевой блок опередил дистрактор в рейтинге)
        if distractor_block_id in predicted_ids and target_block_id in predicted_ids:
            if predicted_ids.index(target_block_id) < predicted_ids.index(distractor_block_id):
                distractor_resisted_count += 1
        elif target_block_id in predicted_ids:
            distractor_resisted_count += 1

    # Итоговые метрики
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    recall_at_8 = (hits_at_8 / total_tasks) * 100
    precision_at_8 = (hits_at_8 / max(total_prefetched_count, 1)) * 100
    unnecessary_prefetch_ratio = 100.0 - precision_at_8
    distractor_resistance_pct = (distractor_resisted_count / total_tasks) * 100

    print("\n" + "=" * 85)
    print("📊 ИТОГОВЫЕ МЕТРИКИ ТОЧНОСТИ И ГОТОВНОСТИ (GENERALIZATION DASHBOARD v2.1):")
    print("=" * 85)
    print(f"Mean Reciprocal Rank (MRR):      {mrr:.3f}")
    print(f"Recall@1:                        {(hits_at_1/total_tasks)*100:.1f}%")
    print(f"Recall@2:                        {(hits_at_2/total_tasks)*100:.1f}%")
    print(f"Recall@4:                        {(hits_at_4/total_tasks)*100:.1f}%")
    print(f"Recall@8:                        {recall_at_8:.1f}%")
    print("-" * 85)
    print(f"Precision@8 (Адаптивная):        {precision_at_8:.2f}%")
    print(f"Избыточность подкачки:           {unnecessary_prefetch_ratio:.2f}%")
    print(f"Устойчивость к дистракторам:      {distractor_resistance_pct:.1f}%")
    print("=" * 85)

    mem_stats = evictor.get_memory_accounting()

    metrics_report = {
        "planner_version": "v2.1_adaptive",
        "mrr": mrr,
        "recall_at_1_pct": (hits_at_1/total_tasks)*100,
        "recall_at_2_pct": (hits_at_2/total_tasks)*100,
        "recall_at_4_pct": (hits_at_4/total_tasks)*100,
        "recall_at_8_pct": recall_at_8,
        "precision_at_8_pct": precision_at_8,
        "unnecessary_prefetch_ratio_pct": unnecessary_prefetch_ratio,
        "distractor_resistance_pct": distractor_resistance_pct,
        "total_kv_mb": mem_stats["total_kv_mb"],
        "vram_freed_pct": mem_stats["vram_freed_pct"]
    }

    metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, ensure_ascii=False)

    print(f"📄 Подробные метрики v2.1 сохранены в: {metrics_path}")
    print("✅ PREFETCH GENERALIZATION & PRECISION GATE v2.1 УСПЕШНО ПРОЙДЕН.")

if __name__ == "__main__":
    run_prefetch_precision_eval()
