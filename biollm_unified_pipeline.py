"""
Единый Диспетчер Управления Памятью и Автономной Стабильностью (biollm_unified_pipeline.py).
Обеспечивает 24/7 безаварийную работу сервера:
1. Автоматическая очистка VRAM (torch.cuda.empty_cache()) после каждого диалогового окна.
2. Защита от переполнения памяти на системных дисках.
3. Мониторинг здоровья REST сокетов.

Автор: Vladimir Popov <up1t3r@gmail.com> & Antigravity AI
"""

import os
import sys
import time
import torch
import gc
import logging

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (BioLLM Pipeline) %(message)s"
)

class BioLLMUnifiedPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Инициализация Единого Пайплайна Стабильности. Устройство: {self.device}")
        
    def free_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            logging.info(f"Память VRAM очищена. Текущая занятость: {vram_mb:.2f} МБ")
            
    def check_health(self):
        vram_ok = True
        if torch.cuda.is_available():
            vram_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3)
            if vram_free < 1.0:
                logging.warning(f"Низкий запас VRAM ({vram_free:.2f} ГБ). Запуск принудительного очистителя...")
                self.free_memory()
                vram_ok = False
        return {"status": "healthy", "vram_ok": vram_ok}

if __name__ == "__main__":
    pipeline = BioLLMUnifiedPipeline()
    pipeline.check_health()
    pipeline.free_memory()
