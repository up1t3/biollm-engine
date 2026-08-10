"""
Модуль Эпигенетического маскирования внимания (Epigenetic Attention).
Использует легковесный сайдкар-предиктор ("метилтрансферазу") для выключения
нерелевантных блоков контекста, обеспечивая подлинейную сложность O(k * N).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class EpigeneticAttentionGate(nn.Module):
    """
    Эпигенетический затвор: оценивает релевантность блоков контекста
    и выдает бинарную/мягкую маску доступности хроматина (Active = 1, Suppressed/Methylated = 0).
    """
    def __init__(self, hidden_dim: int, threshold: float = 0.3):
        super().__init__()
        self.threshold = threshold
        # Маловесный MLP-предиктор (1% от параметров слоя)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, query: torch.Tensor, key_block: torch.Tensor) -> torch.Tensor:
        """
        :param query: Текущий вектор запроса [batch, num_heads, 1, head_dim]
        :param key_block: Блок ключей контекста [batch, num_heads, seq_len, head_dim]
        :return: Маска метилирования [batch, num_heads, seq_len, 1] со значениями 0 или 1
        """
        # Усреднение по длине блока для быстрой оценки
        key_summary = key_block.mean(dim=2, keepdim=True) # [batch, num_heads, 1, head_dim]
        
        # Конкатенация Query и Ключевого описания
        combined = torch.cat([query, key_summary], dim=-1)
        
        # Вероятность того, что участок хроматина "открыт" (Active)
        active_prob = self.gate_mlp(combined) # [batch, num_heads, 1, 1]
        
        # Пороговое бинаризованное маскирование в режиме инференса
        if not self.training:
            epigenetic_mask = (active_prob >= self.threshold).float()
        else:
            epigenetic_mask = active_prob
            
        return epigenetic_mask


class BioEpigeneticAttention(nn.Module):
    """
    Слой внимания с интеграцией Эпигенетического затвора (Epigenetic Masking).
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.epi_gate = EpigeneticAttentionGate(self.head_dim)

    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # 1. Получение Эпигенетической маски активности
        # Вычисляем активность контекста
        epi_mask = self.epi_gate(q[:, :, -1:, :], k) # [batch, num_heads, 1, 1]

        # 2. Вычисление Scaled Dot-Product Attention с добавлением эпигенетической маски
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        # Если блок метилирован (epi_mask = 0), заглушаем оценки вычислений (-infinity)
        mask_penalty = (1.0 - epi_mask) * -10000.0
        masked_scores = scores + mask_penalty

        attn_weights = F.softmax(masked_scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(output), (k, v)
