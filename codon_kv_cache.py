"""
Модуль сжатия KV-Кэша "Codon KV Engine".
Группирует 3 последовательных вектора контекста (триплеты) в 1 кодонный супер-вектор,
сокращая использование видеопамяти VRAM в 3 раза.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

class CodonKVCacheManager(nn.Module):
    """
    Менеджер сжатия кэша Ключей и Значений (KV-Cache) на основе триплетного кодирования.
    """
    def __init__(self, head_dim: int, group_size: int = 3):
        super().__init__()
        self.head_dim = head_dim
        self.group_size = group_size
        
        # Легковесный линейный проектор для агрегации триплетов в 1 кодон
        self.codon_compressor = nn.Linear(head_dim * group_size, head_dim, bias=False)
        # Инициализируем усреднением для стабильного старта
        with torch.no_grad():
            weights = torch.cat([torch.eye(head_dim) / group_size for _ in range(group_size)], dim=1)
            self.codon_compressor.weight.copy_(weights)

    def compress_kv_triplet(self, kv_window: torch.Tensor) -> torch.Tensor:
        """
        Сжимает временное окно из 3 токенов [batch, heads, 3, head_dim] -> [batch, heads, 1, head_dim]
        
        :param kv_window: Тензор KV размерности [batch, num_heads, 3, head_dim]
        :return: Сжатый кодонный вектор [batch, num_heads, 1, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = kv_window.shape
        assert seq_len == self.group_size, f"Ожидалась длина окна {self.group_size}, получена {seq_len}"

        # Конкатенация 3 векторов вдоль размерности фичей
        # [batch, num_heads, 3 * head_dim]
        concatenated = kv_window.transpose(2, 3).reshape(batch_size, num_heads, head_dim * self.group_size)
        
        # Сжатие в 1 кодонный вектор
        codon_vector = self.codon_compressor(concatenated).unsqueeze(2) # [batch, num_heads, 1, head_dim]
        return codon_vector

    def append_to_cache(self, current_cache: Optional[torch.Tensor], new_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Добавляет новые состояния в кэш. Когда накапливается 3 состояния, они упаковываются в кодон.
        
        :param current_cache: Текущий сжатый кэш [batch, num_heads, num_codons, head_dim] или None
        :param new_states: Новые KV состояния [batch, num_heads, seq_len, head_dim]
        :return: (обновленный_сжатый_кэш, остаточный_буфер)
        """
        if current_cache is None:
            full_seq = new_states
        else:
            full_seq = torch.cat([current_cache, new_states], dim=2)

        batch_size, num_heads, total_len, head_dim = full_seq.shape
        num_codons = total_len // self.group_size
        remainder_len = total_len % self.group_size

        if num_codons > 0:
            # Выделяем часть, кратную 3
            codon_input = full_seq[:, :, :num_codons * self.group_size, :]
            # Реформируем для батчевого прохода через сжиматель
            codon_input_reshaped = codon_input.view(batch_size, num_heads, num_codons, self.group_size, head_dim)
            
            compressed_list = []
            for i in range(num_codons):
                triplet = codon_input_reshaped[:, :, i, :, :] # [batch, heads, 3, head_dim]
                compressed = self.compress_kv_triplet(triplet)
                compressed_list.append(compressed)
                
            compressed_cache = torch.cat(compressed_list, dim=2) # [batch, heads, num_codons, head_dim]
        else:
            compressed_cache = torch.empty((batch_size, num_heads, 0, head_dim), device=new_states.device, dtype=new_states.dtype)

        remainder_cache = full_seq[:, :, num_codons * self.group_size:, :] if remainder_len > 0 else None

        return compressed_cache, remainder_cache
