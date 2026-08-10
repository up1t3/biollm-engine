"""
Модуль векторной индексации блоков BlockRetrievalIndex.
Обеспечивает векторный семантический поиск по 1024+ блокам через матричное умножение PyTorch
со скоростью менее 1 мс.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional

class BlockRetrievalIndex:
    """
    Векторный индекс семантического поиска по блокам KV-памяти.
    """
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.block_ids: List[int] = []
        self.embeddings: Optional[torch.Tensor] = None # Tensor [N, D]

    def add_or_update_block(self, block_id: int, embedding: torch.Tensor):
        """
        Добавляет или обновляет эмбеддинг блока в векторном индексе.
        """
        # Сжатие любого входного тензора в единый 1D вектор размерности D
        flat_emb = embedding.detach().float().flatten()
        if flat_emb.numel() > self.embedding_dim:
            flat_emb = flat_emb[:self.embedding_dim]
        elif flat_emb.numel() < self.embedding_dim:
            flat_emb = F.pad(flat_emb, (0, self.embedding_dim - flat_emb.numel()))

        emb = F.normalize(flat_emb.unsqueeze(0), dim=-1) # [1, D]
        
        if self.embeddings is None:
            self.embeddings = emb
            self.block_ids = [block_id]
        else:
            if block_id in self.block_ids:
                idx = self.block_ids.index(block_id)
                self.embeddings[idx] = emb[0]
            else:
                self.embeddings = torch.cat([self.embeddings, emb], dim=0)
                self.block_ids.append(block_id)

    def search_top_k(self, query_embedding: torch.Tensor, top_k: int = 8) -> List[Tuple[int, float]]:
        """
        Векторизованный матричный поиск Top-K наиболее релевантных блоков.
        """
        if self.embeddings is None or len(self.block_ids) == 0:
            return []

        flat_q = query_embedding.detach().float().flatten()
        if flat_q.numel() > self.embedding_dim:
            flat_q = flat_q[:self.embedding_dim]
        elif flat_q.numel() < self.embedding_dim:
            flat_q = F.pad(flat_q, (0, self.embedding_dim - flat_q.numel()))

        q_emb = F.normalize(flat_q.unsqueeze(0), dim=-1) # [1, D]
        
        # Матричное скалярное произведение тензоров [1, D] @ [N, D].T -> [1, N]
        sim_scores = torch.matmul(q_emb, self.embeddings.T).squeeze(0) # [N]
        
        k = min(top_k, len(self.block_ids))
        top_scores, top_indices = torch.topk(sim_scores, k=k)

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append((self.block_ids[idx], score))

        return results
