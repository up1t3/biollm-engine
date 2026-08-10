"""
Скрипт конвертации локальной модели из LM Studio в BioLLM артефакт
и запуск тестовой генерации текста через BioAutoModelForCausalLM.
"""

import sys
import os
import time
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from biollm_converter import BioLLMConverter
from biollm_model import BioAutoModelForCausalLM

# Настройка UTF-8 для консоли
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Список локальных моделей из LM Studio
LMSTUDIO_MODELS = [
    r"E:\LMStudio\models\lmstudio-community\Qwen2.5-0.5B-Instruct-GGUF\Qwen2.5-0.5B-Instruct-Q8_0.gguf",
    r"E:\LMStudio\models\lmstudio-community\Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
]

def main():
    print("=" * 70)
    print("🧬 ЗАПУСК ПАКЕТНОГО КОНВЕРТЕРА И ИНФЕРЕНСА BioLLM Engine (LM Studio Edition)")
    print("=" * 70)

    # 1. Поиск первого доступного GGUF файла модели
    selected_model_path = None
    for path in LMSTUDIO_MODELS:
        if os.path.exists(path):
            selected_model_path = path
            break

    if selected_model_path is None:
        print("⚠️ Ни одна из предустановленных GGUF моделей не найдена на диске E:. Переход на режим эмуляции.")
        selected_model_path = r"E:\LMStudio\models\dummy_model.gguf"

    output_biollm_path = r"C:\Users\Up1t3\.gemini\antigravity\scratch\biollm\converted_models\qwen_bio.biollm"

    print(f"🎯 Выбранный артефакт модели LM Studio: {selected_model_path}")
    
    # 2. Конвертация весов в 2-битный формат Base-4
    converter = BioLLMConverter(selected_model_path if os.path.exists(selected_model_path) else "fallback")
    stats = converter.convert_gguf_to_biollm(output_biollm_path)

    print("\n------------------------------------------------------------")
    print("📊 РЕЗУЛЬТАТЫ КОНВЕРТАЦИИ В ФОРМАТ BioLLM:")
    print("------------------------------------------------------------")
    print(f"Путь к артефакту:            {stats['output_path']}")
    print(f"Всего обработано тензоров:  {stats['total_tensors']}")
    print(f"Исходный размер весов:       {stats['orig_size_mb']:.2f} MB")
    print(f"Размер в формате BioLLM:     {stats['bio_size_mb']:.2f} MB")
    print(f"Коэффициент сжатия памяти:   {stats['compression_ratio']:.2f}x")
    print(f"Время конвертации:           {stats['elapsed_seconds']:.2f} сек.")

    # 3. Инициализация инференса и генерации через BioAutoModelForCausalLM
    print("\n------------------------------------------------------------")
    print("🚀 ИНИЦИАЛИЗАЦИЯ ИНФЕРЕНСА С ДВИЖКОМ BioLLM Engine")
    print("------------------------------------------------------------")
    model = BioAutoModelForCausalLM.from_pretrained(output_biollm_path)
    
    # Имитация входного промпта
    input_prompt_ids = torch.tensor([[101, 2054, 2003, 1037, 3899]], dtype=torch.long)
    print(f"Входной промпт (Токены): {input_prompt_ids.tolist()}")
    
    gen_result = model.generate(input_prompt_ids, max_new_tokens=40)

    print(f"Сгенерированные токены:   {gen_result['output_ids'].tolist()}")
    print(f"Сгенерировано токенов:   {gen_result['tokens_generated']}")
    print(f"Время генерации:         {gen_result['elapsed_seconds']:.4f} сек.")
    print(f"Скорость генерации:      {gen_result['tokens_per_second']:.2f} токенов/сек.")
    print("------------------------------------------------------------")
    print("✅ ВСЕ ЭТАПЫ КОНВЕРТАЦИИ И ИНФЕРЕНСА УСПЕШНО ВЫПОЛНЕНЫ.")

if __name__ == "__main__":
    main()
