"""
Скрипт Скачивания Флагманской Модели Qwen2.5-72B-Instruct.IQ2_XS.gguf на Диск E: (download_qwen72b_iq2_xs.py).

Выполняет прямое скачивание 2-битной сжатой GGUF модели Qwen2.5-72B (IQ2_XS)
из HuggingFace на накопитель E:\biollm_models\qwen2.5_72b_iq2_xs.gguf:
- Размер файла: ~23 ГБ (Предельная 2-битная точность IQ2_XS).
- Идеально под 1x 24GB GPU (RTX 3090 / 4090).

Автор: Vladimir Popov <up1t3r@gmail.com> & Antigravity AI
"""

import os
import sys
import time
from huggingface_hub import hf_hub_download

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

TARGET_DIR = "E:/biollm_models"
REPO_ID = "bartowski/Qwen2.5-72B-Instruct-GGUF"
FILENAME = "Qwen2.5-72B-Instruct-IQ2_XS.gguf"

def download_iq2_xs():
    print("=" * 85)
    print("🚀 СТАРТ СКАЧИВАНИЯ ФЛАГМАНСКОЙ 72B МОДЕЛИ Qwen2.5-72B-Instruct (IQ2_XS GGUF)")
    print("=" * 85)
    print(f"📁 Накопитель: E:\\biollm_models\\{FILENAME}")
    print(f"🌐 Источник: https://huggingface.co/{REPO_ID}")
    print("------------------------------------------------------------")
    
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    try:
        t0 = time.time()
        file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=TARGET_DIR,
            local_dir_use_symlinks=False
        )
        t_elapsed = time.time() - t0
        print("------------------------------------------------------------")
        print(f"🏆 ФЛАГМАНСКАЯ МОДЕЛЬ Qwen2.5-72B IQ2_XS УСПЕШНО СКАЧАНА НА ДИСК E:")
        print(f"  • Путь к файлу: {file_path}")
        print(f"  • Время скачивания: {t_elapsed/60:.2f} минут")
        print("=================================================================")
    except Exception as e:
        print(f"⚠️ Ошибка при скачивании из {REPO_ID}: {e}")
        # Пробуем fallback репозиторий
        try:
            print("🔄 Пробуем альтернативное зеркало unsloth/Qwen2.5-72B-Instruct-GGUF...")
            file_path = hf_hub_download(
                repo_id="unsloth/Qwen2.5-72B-Instruct-GGUF",
                filename="Qwen2.5-72B-Instruct-Q2_K.gguf",
                local_dir=TARGET_DIR,
                local_dir_use_symlinks=False
            )
            print(f"✅ Успешно скачано с зеркала Unsloth: {file_path}")
        except Exception as ex2:
            print(f"❌ Запасной вызов: {ex2}")

if __name__ == "__main__":
    download_iq2_xs()
