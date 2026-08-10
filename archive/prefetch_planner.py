"""
Модуль предсказательной подкачки памяти Prefetch Planner Engine.
Использует внимание, семантическое сходство и приоритеты для заблаговременной
асинхронной подкачки KV-блоков из CPU RAM в GPU VRAM перед вычислениями.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

class PrefetchPlanner:
    """
    Планировщик упреждающей подкачки KV-блоков.
    """
    def __init__(self, top_k: int = 4, alpha_attention: float = 0.4, beta_semantic: float = 0.4, gamma_recency: float = 0.2):
        self.top_k = top_k
        self.alpha_attention = alpha_attention
        self.beta_semantic = beta_semantic
        self.gamma_recency = gamma_recency

    def predict_next_blocks(
        self,
        query_vector: torch.Tensor,
        evicted_cpu_blocks: List[Dict[str, Any]],
        attention_history: Optional[torch.Tensor] = None
    ) -> List[int]:
        """
        Прогнозирует топ-K ID блоков, которые понадобятся модели на следующих шагах генерации.
        """
        if not evicted_cpu_blocks:
            return []

        scored_blocks = []
        
        for idx, block in enumerate(evicted_cpu_blocks):
            block_id = block["block_id"]
            
            # 1. Оценка внимания (Attention score)
            attn_score = 0.5
            if attention_history is not None and block_id < attention_history.shape[-1]:
                attn_score = attention_history[..., block_id].mean().item()

            # 2. Семантическая схожесть с текущим запросом (Semantic Score)
            block_tensor = block["kv_tensor"].float()
            # Проекция через скалярное произведение норм
            similarity = torch.cosine_similarity(
                query_vector.mean(dim=(0, 1, 2)),
                block_tensor.mean(dim=(0, 1, 2)),
                dim=-1
            ).item()
            semantic_score = max(0.0, (similarity + 1.0) / 2.0)

            # 3. Показатель свежести (Recency score)
            recency_score = 1.0 / (idx + 1.0)

            # Итоговый комбинированный балл приоритета префетча
            total_score = (
                self.alpha_attention * attn_score +
                self.beta_semantic * semantic_score +
                self.gamma_recency * recency_score
            )

            scored_blocks.append((block_id, total_score))

        # Ранжирование блоков и отбор TOP-K
        scored_blocks.sort(key=lambda x: x[1], reverse=True)
        predicted_ids = [b[0] for b in scored_blocks[:self.top_k]]
        return predicted_ids
