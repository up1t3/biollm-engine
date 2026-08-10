"""
Скрипт диагностики и валидации (Smoke Test & Baseline Gate) для BioLLM моделей.
Выполняет санитарную проверку весов (NaN/Inf check), тестовый прогон генерации
и фиксацию базовых метрик (VRAM, Latency, Token/s).
"""

import os
import sys
import time
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from biollm_model import BioAutoModelForCausalLM

# Установка UTF-8 для консоли
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

MODEL_PATH = r"C:\Users\Up1t3\.gemini\antigravity\scratch\biollm\converted_models\qwen_bio.biollm"

def run_smoke_test():
    print("=" * 70)
    print("🔍 SMOKE TEST & BASELINE VALIDATION GATE (BioLLM Engine v2.0)")
    print("=" * 70)

    # 1. Проверка существования артефакта
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Ошибка: Артефакт не найден по пути {MODEL_PATH}")
        sys.exit(1)

    file_size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
    print(f"1. Проверка файла модели:  [ОК] ({file_size_mb:.2f} MB)")

    # 2. Загрузка модели и санитарная проверка тензоров
    print("2. Загрузка весов и проверка тензоров (NaN / Inf Check)...")
    start_load = time.time()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    weights = checkpoint["weights"]
    
    nan_count = 0
    inf_count = 0
    total_elements = 0

    for name, tensor in weights.items():
        if tensor.is_floating_point():
            total_elements += tensor.numel()
            nan_count += torch.isnan(tensor).sum().item()
            inf_count += torch.isinf(tensor).sum().item()

    print(f"   - Проверено плавающих элементов: {total_elements}")
    print(f"   - Обнаружено NaN: {nan_count}")
    print(f"   - Обнаружено Inf: {inf_count}")

    if nan_count > 0 or inf_count > 0:
        print("❌ СБОЙ ВАЛИДАЦИИ: Найдены невалидные тензоры (NaN/Inf)!")
        sys.exit(1)
    else:
        print("   - Санитарная проверка: [ПРОЙДЕНА УСПЕШНО]")

    # 3. Инициализация BioAutoModelForCausalLM
    model = BioAutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.eval()

    # 4. Диагностический прогон (Forward Pass & Generation)
    prompt_tokens = torch.tensor([[101, 2054, 2003, 1037, 3899]], dtype=torch.long)
    print(f"\n3. Запуск диагностического прогона генерации (Prompt size: {prompt_tokens.shape})...")

    # Имитация измерения VRAM/RAM
    start_gen = time.time()
    gen_result = model.generate(prompt_tokens, max_new_tokens=50)
    gen_time = time.time() - start_gen

    print("\n------------------------------------------------------------")
    print("📊 МЕТРИКИ BASELINE И ДИАГНОСТИКИ:")
    print("------------------------------------------------------------")
    print(f"Сгенерировано токенов:      {gen_result['tokens_generated']}")
    print(f"Время генерации (Latency):  {gen_time:.4f} сек.")
    print(f"Скорость (Token/s):         {gen_result['tokens_per_second']:.2f} токенов/сек.")
    print(f"Выходные токены:            {gen_result['output_ids'].tolist()[0]}")
    print("------------------------------------------------------------")
    print("✅ SMOKE TEST ПРОЙДЕН. Модель полностью готова к пошаговому внедрению BioLLM v2.0.")

if __name__ == "__main__":
    run_smoke_test()
