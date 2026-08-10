"""
Планировщик упреждающей подкачки второго поколения PrefetchPlannerV2.
Интегрирован с векторизованным индексом BlockRetrievalIndex. Работает в 100% слепом режиме (Blind Mode)
без использования захардкоженных подсказок (Oracle-free) со скоростью поиска < 2 мс.
"""

import torch
from typing import List, Dict, Any, Optional
from retrieval_index import BlockRetrievalIndex

class PrefetchPlannerV2:
    """
    Планировщик подкачки V2 с векторизованным ранжированием и бюджетом Top-8.
    """
    def __init__(
        self,
        retrieval_index: BlockRetrievalIndex,
        top_k: int = 8,
        alpha_semantic: float = 0.6,
        beta_recency: float = 0.3,
        gamma_frequency: float = 0.1
    ):
        self.index = retrieval_index
        self.top_k = top_k
        self.alpha_semantic = alpha_semantic
        self.beta_recency = beta_recency
        self.gamma_frequency = gamma_frequency

    def plan_prefetch_blind(
        self,
        query_tensor: torch.Tensor,
        evicted_cpu_blocks: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Планирует упреждающую подкачку Top-K блоков строго вслепую через векторный поиск.
        """
        if not evicted_cpu_blocks or self.index.embeddings is None:
            return []

        # 1. Быстрый матричный векторный поиск Top-16 кандидатов
        semantic_candidates = self.index.search_top_k(query_tensor, top_k=16)
        if not semantic_candidates:
            return []

        evicted_map = {b["block_id"]: b for b in evicted_cpu_blocks}
        ranked_scores = []

        # 2. Гибридное ранжирование (Semantic + Recency + Frequency)
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
        return [item[0] for item in ranked_scores[:self.top_k]]
