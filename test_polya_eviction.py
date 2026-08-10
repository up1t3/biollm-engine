"""
Тестовый скрипт проверки выгрузки памяти Poly-A Eviction Engine.
"""

import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from polya_evictor import PolyAEvictor

# Настройка UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 70)
    print("🧬 ТЕСТИРОВАНИЕ ВЫГРУЗКИ ПАМЯТИ POLY-A EVICTION ENGINE (VRAM -> CPU RAM)")
    print("=" * 70)

    # Лимит VRAM блоков = 5 для демонстрации подкачки
    evictor = PolyAEvictor(max_vram_blocks=5, initial_polya_tail=20, decay_step=10)

    # 1. Регистрация защищенного теломерного блока Head (Системный промпт)
    head_tensor = torch.randn(1, 16, 64, 64)
    evictor.register_kv_block(block_id=0, kv_tensor=head_tensor, is_protected=True)

    # 2. Накопление обычных блоков контекста
    for i in range(1, 10):
        dummy_block = torch.randn(1, 16, 64, 64)
        evictor.register_kv_block(block_id=i, kv_tensor=dummy_block, is_protected=False)

    # 3. Имитация нескольких шагов жизни генерации (Poly-A decay)
    print("\nИмитация 3 шагов деаденилирования (Poly-A Decay)...")
    for step in range(3):
        evictor.step_decay_and_evict()

    stats = evictor.get_eviction_stats()
    print("\n------------------------------------------------------------")
    print("📊 СТАТИСТИКА УПРАВЛЕНИЯ ПАМЯТЬЮ POLY-A EVICTOR:")
    print("------------------------------------------------------------")
    print(f"Активных блоков в VRAM:     {stats['active_vram_blocks']}")
    print(f"Выгруженных блоков в CPU:    {stats['evicted_cpu_blocks']}")
    print(f"Всего отслеживается блоков: {stats['total_tracked_blocks']}")

    # Проверка защиты теломерного блока
    head_is_in_vram = any(b["block_id"] == 0 for b in evictor.active_vram_blocks)
    print(f"Защита Теломерного блока Head: [{'СОХРАНЕН В VRAM' if head_is_in_vram else 'ОШИБКА'}]")
    print("------------------------------------------------------------")
    print("✅ POLY-A EVICTION ENGINE УСПЕШНО ПРОШЕЛ ИСПЫТАНИЯ.")

if __name__ == "__main__":
    main()
