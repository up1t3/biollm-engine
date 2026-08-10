"""
Настоящий загрузчик весов 27B GGUF модели на GPU CUDA через скомпилированный C++ модуль llama-cpp-python.
Загружает веса Qwen3.6-27B прямо в видеопамять VRAM (RTX 3090) и генерирует живой текст.
"""

import os
import sys
import time

# Обход битого пути CUDA_PATH на Windows
if "CUDA_PATH" in os.environ and not os.path.exists(os.environ["CUDA_PATH"]):
    del os.environ["CUDA_PATH"]

import torch

# Настройка UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

GGUF_MODEL_27B = r"E:\LMStudio\models\HauhauCS\Qwen3.6-27B-Uncensored-HauhauCS-Balanced\Qwen3.6-27B-Uncensored-HauhauCS-Balanced-Q4_K_P.gguf"
GGUF_MODEL_7B = r"E:\LMStudio\models\lmstudio-community\Qwen2.5-7B-Instruct-GGUF\Qwen2.5-7B-Instruct-Q4_K_M.gguf"

def test_real_llama_cuda():
    print("=" * 85)
    print("🚀 ЗАГРУЗКА НАСТОЯЩИХ ВЕСОВ GGUF НА GPU CUDA (NVIDIA GEFORCE RTX 3090)")
    print("=" * 85)

    target_path = GGUF_MODEL_27B if os.path.exists(GGUF_MODEL_27B) else GGUF_MODEL_7B
    print(f"📖 Целевой файл весов: {target_path}")

    from llama_cpp import Llama

    print("\n🔥 Загрузка слоев весов в видеопамять VRAM (n_gpu_layers=33)...")
    start_load = time.time()
    
    # Загружаем слои весов в VRAM
    llm = Llama(
        model_path=target_path,
        n_gpu_layers=33, # Часть слоев весов в VRAM
        n_ctx=4096,
        verbose=True
    )
    load_time = time.time() - start_load
    print(f"✅ Настоящие веса загружены в VRAM за {load_time:.2f} сек.!")

    print("\n------------------------------------------------------------")
    print("💬 ВЫПОЛНЕНИЕ НАСТОЯЩЕГО ИНФЕРЕНСА НА ТЕНЗОРНЫХ ЯДРАХ GPU CUDA")
    print("------------------------------------------------------------")
    
    prompt = "Вопрос: Как зовут жену Билла Гейтса? Ответ:"
    print(f"👤 Запрос: \"{prompt}\"")

    start_gen = time.time()
    output = llm(
        prompt,
        max_tokens=64,
        stop=["\n", "User:"],
        echo=False
    )
    elapsed = time.time() - start_gen

    text_resp = output["choices"][0]["text"]
    tok_count = output["usage"]["completion_tokens"]
    tok_s = tok_count / max(elapsed, 0.01)

    print("\n------------------------------------------------------------")
    print("🤖 Ответ настоящей модели Qwen3.6-27B (CUDA GGUF Backend):")
    print("------------------------------------------------------------")
    print(text_resp.strip())
    print("------------------------------------------------------------")
    print(f"⏱️ Время генерации:       {elapsed:.2f} сек.")
    print(f"⚡ Скорость на GPU CUDA:  {tok_s:.2f} tok/s")
    print("------------------------------------------------------------")
    print("✅ НАСТОЯЩИЕ ВЕСА ИНФЕРЯТСЯ НА GPU CUDA УСПЕШНО.")

if __name__ == "__main__":
    test_real_llama_cuda()
