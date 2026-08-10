"""
Модуль Полимеразной Коррекции Ошибок (Polymerase Proofreader Engine).
Выполняет частичный или полный пересчет слоев в точности FP16
только при обнаружении аномальных всплесков квантового шума в 2-битных весах Base-4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

class PolymeraseProofreadLinear(nn.Module):
    """
    Линейный слой с автоматической полимеразной коррекцией ошибок (Proofreading).
    """
    def __init__(
        self,
        base4_layer: nn.Module,
        spike_threshold: float = 5.0,
        max_partial_fallback_ratio: float = 0.20
    ):
        super().__init__()
        self.base4_layer = base4_layer
        self.spike_threshold = spike_threshold
        self.max_partial_fallback_ratio = max_partial_fallback_ratio
        
        # Резервные веса FP16 для спасения точности (хранятся во 2-м потоке)
        in_features = base4_layer.in_features
        out_features = base4_layer.out_features
        self.register_buffer("rescue_weight", torch.randn(out_features, in_features, dtype=torch.float16) * 0.02)
        
        # Статистика срабатывания fallback
        self.total_calls = 0
        self.partial_fallbacks = 0
        self.full_fallbacks = 0

    def evaluate_health(self, y: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет аномальность токенов на основе отклонения их нормы ||x||2 от средней нормы батча.
        """
        yf = y.float()
        token_norms = yf.norm(dim=-1) # [batch, seq_len]
        mean_norm = token_norms.mean()
        if token_norms.numel() > 1:
            std_norm = token_norms.std(unbiased=False).clamp_min(1e-5)
        else:
            std_norm = torch.tensor(1.0, device=yf.device)
        
        # Токены, чья норма отклоняется более чем на N сигм
        bad_tokens_mask = token_norms > (mean_norm + self.spike_threshold * std_norm)
        return bad_tokens_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.total_calls += 1
        
        # 1. Быстрый проход через 2-битный Base-4 слой
        y_base4 = self.base4_layer(x)
        
        # 2. Проверка аномальности активаций
        bad_mask = self.evaluate_health(y_base4) # Bool tensor [batch, seq_len]
        
        if not bad_mask.any():
            return y_base4

        bad_ratio = bad_mask.float().mean().item()

        # 3. Режим спасения (Rescue Mode)
        rescue_w = self.rescue_weight.to(x.device, dtype=x.dtype)

        if bad_ratio <= self.max_partial_fallback_ratio:
            # Частичный Proofreading: пересчитываем в FP16 ТОЛЬКО аномальные токены
            self.partial_fallbacks += 1
            y_fixed = y_base4.clone()
            
            x_bad = x[bad_mask] # [num_bad, in_features]
            y_bad = F.linear(x_bad, rescue_w)
            y_fixed[bad_mask] = y_bad
            return y_fixed
        else:
            # Массовая аномалия: полный пересчет слоя в FP16
            self.full_fallbacks += 1
            y_fp16 = F.linear(x, rescue_w)
            return y_fp16

    def get_proofreader_stats(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "partial_fallbacks": self.partial_fallbacks,
            "full_fallbacks": self.full_fallbacks,
            "fallback_rate_pct": ((self.partial_fallbacks + self.full_fallbacks) / max(self.total_calls, 1)) * 100
        }
