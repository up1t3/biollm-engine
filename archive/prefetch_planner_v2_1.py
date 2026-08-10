"""
Планировщик упреждающей подкачки версии v2.1 (Adaptive Top-K & Confidence Budgeting).
Динамически регулирует бюджет подкачки K (от 2 до 8) в зависимости от уверенности (Score Gap),
снижая избыточный трафик PCIe и экономя VRAM.
"""

import torch
from typing import List, Dict, Any, Tuple
from retrieval_index import BlockRetrievalIndex

class PrefetchPlannerV21:
    """
    Адаптивный планировщик V2.1 с уверенностным бюджетом (Confidence Budgeting).
    """
    def __init__(
        self,
        retrieval_index: BlockRetrievalIndex,
        min_k: int = 2,
        max_k: int = 8,
        confidence_gap_threshold: float = 0.15,
        alpha_semantic: float = 0.6,
        beta_recency: float = 0.3,
        gamma_frequency: float = 0.1
    ):
        self.index = retrieval_index
        self.min_k = min_k
        self.max_k = max_k
        self.confidence_gap_threshold = confidence_gap_threshold
        self.alpha_semantic = alpha_semantic
        self.beta_recency = beta_recency
        self.gamma_frequency = gamma_frequency

    def plan_prefetch_adaptive(
        self,
        query_tensor: torch.Tensor,
        evicted_cpu_blocks: List[Dict[str, Any]]
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Адаптивно планирует подкачку Top-K с динамическим определением K по уверености.
        """
        if not evicted_cpu_blocks or self.index.embeddings is None:
            return [], {"dynamic_k": self.min_k, "confidence_gap": 0.0}

        # 1. Быстрый матричный векторный поиск Top-16
        semantic_candidates = self.index.search_top_k(query_tensor, top_k=16)
        if not semantic_candidates:
            return [], {"dynamic_k": self.min_k, "confidence_gap": 0.0}

        evicted_map = {b["block_id"]: b for b in evicted_cpu_blocks}
        ranked_scores = []

        # 2. Ранжирование
        for block_id, sem_score in semantic_candidates:
            if block_id in evicted_map:
                block_meta = evicted_map[block_id]
                recency = 1.0 / (block_meta.get("access_count", 1) + 1.0)
                freq = min(1.0, block_meta.get("access_count", 1) / 10.0)

                final_score = (
                    self.alpha_semantic * sem_score +
                    self.beta_recency * recency +
                    self.gamma_frequency * freq
                )
                ranked_scores.append((block_id, final_score))

        ranked_scores.sort(key=lambda x: x[1], reverse=True)

        if len(ranked_scores) < 2:
            dynamic_k = self.min_k
            gap = 0.0
        else:
            gap = ranked_scores[0][1] - ranked_scores[1][1]
            # Если разрыв уверенности высокий -> достаточно min_k (top-2), иначе расширяем до max_k (top-8)
            if gap >= self.confidence_gap_threshold:
                dynamic_k = self.min_k
            else:
                dynamic_k = self.max_k

        selected_ids = [item[0] for item in ranked_scores[:dynamic_k]]
        
        meta = {
            "dynamic_k": dynamic_k,
            "confidence_gap": gap,
            "top_score": ranked_scores[0][1] if ranked_scores else 0.0
        }
        return selected_ids, meta
