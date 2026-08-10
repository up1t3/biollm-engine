"""
Исследовательский Модуль Telomeric Head/Tail Protection Layer (biollm_telomeric_protection.py).

Обеспечивает абсолютную маскировочную защиту ("Теломеры"):
- Telomeric Head Anchor (Защита первых 512 токенов системного промпта).
- Telomeric Tail Anchor (Защита последних 256 токенов текущего генеративного контекста).

Теломерные блоки имеют максимальный приоритет иммунитета (Eviction-Proof Masking)
и ни при каких условиях не удаляются из KV-кэша.
"""

import torch

class TelomericProtectionLayer:
    def __init__(self, head_size=512, tail_size=256):
        self.head_size = head_size
        self.tail_size = tail_size

    def compute_telomeric_mask(self, total_seq_len: int, device='cpu'):
        """
        Генерирует бинарную маску теломерной защиты размера [total_seq_len].
        True = Токен защищен теломерой (Head или Tail) и НЕ МОЖЕТ быть вытеснен.
        False = Токен находится в промежуточной Poly-A зоне и может быть сжат.
        """
        mask = torch.zeros(total_seq_len, dtype=torch.bool, device=device)
        
        if total_seq_len <= (self.head_size + self.tail_size):
            # Если вся последовательность меньше теломерных границ — защищаем 100%
            mask[:] = True
            return mask
            
        # Защита Head-теломеры (системный промпт)
        mask[:self.head_size] = True
        
        # Защита Tail-теломеры (текущий активный контекст генерации)
        mask[-self.tail_size:] = True
        
        return mask

    def get_protected_indices(self, total_seq_len: int, device='cpu'):
        """
        Возвращает тензор индексов защищенных токенов.
        """
        mask = self.compute_telomeric_mask(total_seq_len, device)
        return torch.nonzero(mask, as_tuple=True)[0]

if __name__ == "__main__":
    print("🧪 Тестирование Telomeric Protection Layer...")
    telomere = TelomericProtectionLayer(head_size=512, tail_size=256)
    
    seq_len = 10000 # 10,000 токенов
    mask = telomere.compute_telomeric_mask(seq_len)
    
    protected_count = mask.sum().item()
    evictable_count = seq_len - protected_count
    
    print(f"📊 Общая длина контекста:      {seq_len} токенов")
    print(f"🛡️ Защищено Теломерами (Head+Tail): {protected_count} токенов ({protected_count/seq_len*100:.1f}%)")
    print(f"🧹 Доступно для Poly-A вытеснения:  {evictable_count} токенов ({evictable_count/seq_len*100:.1f}%)")
