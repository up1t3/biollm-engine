"""
Скрипт конвертации промышленной 7B/8B модели (Qwen2.5-7B или Llama-3.1-8B)
в 4-ричный биологический кодек BioLLM v3.0 (.biollm).
"""

import os
import sys
import time
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from biollm_converter import BioLLMConverter

# Настройка UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Исходная модель на диске E:
GGUF_MODEL_PATH = r"E:\LMStudio\models\bartowski\Qwen2.5-7B-Instruct-GGUF\Qwen2.5-7B-Instruct-Q4_K_M.gguf"
OUTPUT_BIO_PATH = r"C:\Users\Up1t3\.gemini\antigravity\scratch\biollm\converted_models\qwen7b_bio.biollm"

def convert_7b_model():
    print("=" * 85)
    print("🧬 ИНИЦИАЛИЗАЦИЯ КОНВЕРТАЦИИ 7B МОДЕЛИ (QWEN2.5-7B-INSTRUCT)")
    print("=" * 85)

    if not os.path.exists(GGUF_MODEL_PATH):
        print(f"❌ Модель не найдена по пути: {GGUF_MODEL_PATH}")
        sys.exit(1)

    input_size_mb = os.path.getsize(GGUF_MODEL_PATH) / 1024 / 1024
    print(f"📦 Исходный файл GGUF: {GGUF_MODEL_PATH} ({input_size_mb:.2f} MB)")

    converter = BioLLMConverter(GGUF_MODEL_PATH)
    
    start_time = time.time()
    print("\n⏳ Выполняется Base-4 квантование весов 7B модели...")
    metadata = converter.convert_gguf_to_biollm(OUTPUT_BIO_PATH)
    elapsed = time.time() - start_time

    output_size_mb = os.path.getsize(OUTPUT_BIO_PATH) / 1024 / 1024
    compression_ratio = input_size_mb / max(output_size_mb, 1.0)

    print("\n" + "=" * 85)
    print("📊 ИТОГИ КОНВЕРТАЦИИ 7B МОДЕЛИ В BioLLM v3.0:")
    print("=" * 85)
    print(f"Время конвертации:                {elapsed:.2f} сек.")
    print(f"Размер исходной модели GGUF:     {input_size_mb:.2f} MB")
    print(f"Размер итогового файла .biollm:  {output_size_mb:.2f} MB")
    print(f"Коэффициент сжатия весов:         {compression_ratio:.2f}x")
    print(f"Слоев конвертировано:            {metadata.get('num_layers', 28)}")
    print("=" * 85)
    print(f"✅ Файл 7B модели успешно сохранен: {OUTPUT_BIO_PATH}")

if __name__ == "__main__":
    convert_7b_model()
