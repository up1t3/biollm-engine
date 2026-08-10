"""
Прямой тест инференса весов 7B GGUF на GPU CUDA (NVIDIA GeForce RTX 3090).
Загружает веса Qwen2.5-7B в видеопамять VRAM и генерирует живой текст.
"""

import os
import sys
import time

if "CUDA_PATH" in os.environ and not os.path.exists(os.environ["CUDA_PATH"]):
    del os.environ["CUDA_PATH"]

from llama_cpp import Llama

# Настройка UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

GGUF_MODEL_7B = r"E:\LMStudio\models\lmstudio-community\Qwen2.5-7B-Instruct-GGUF\Qwen2.5-7B-Instruct-Q4_K_M.gguf"

def run_real_7b_cuda_test():
    print("=" * 85)
    print("🚀 НАСТОЯЩИЙ ИНФЕРЕНС ВЕСОВ QWEN2.5-7B В ВИДЕОПАМЯТИ VRAM (RTX 3090)")
    print("=" * 85)

    if not os.path.exists(GGUF_MODEL_7B):
        print(f"❌ Модель не найдена: {GGUF_MODEL_7B}")
        return

    print("🔥 Загрузка всех 28 слоев весов 7B в VRAM (n_gpu_layers=99)...")
    start_load = time.time()
    
    llm = Llama(
        model_path=GGUF_MODEL_7B,
        n_gpu_layers=99, # Загружаем ВСЕ слои весов прямо в VRAM GPU!
        n_ctx=4096,
        verbose=False
    )
    load_time = time.time() - start_load
    print(f"✅ Настоящие веса 7B загружены в VRAM за {load_time:.2f} сек.!")

    messages = [
        {"role": "system", "content": "Ты профессиональный ИИ-ассистент BioLLM."},
        {"role": "user", "content": "Как зовут бывшую жену Билла Гейтса и чем она известна?"}
    ]

    print("\n------------------------------------------------------------")
    print("💬 ВЫПОЛНЕНИЕ НАСТОЯЩЕГО ИНФЕРЕНСА НА ТЕНЗОРНЫХ ЯДРАХ GPU CUDA")
    print("------------------------------------------------------------")
    print(f"👤 Запрос: \"{messages[1]['content']}\"")

    start_gen = time.time()
    resp = llm.create_chat_completion(
        messages=messages,
        max_tokens=100,
        temperature=0.7
    )
    elapsed = time.time() - start_gen

    ans = resp["choices"][0]["message"]["content"]
    tok_count = resp["usage"]["completion_tokens"]
    tok_s = tok_count / max(elapsed, 0.01)

    print("\n------------------------------------------------------------")
    print("🤖 Настоящий генеративный ответ модели Qwen (CUDA VRAM):")
    print("------------------------------------------------------------")
    print(ans.strip())
    print("------------------------------------------------------------")
    print(f"⏱️ Время генерации:       {elapsed:.2f} сек.")
    print(f"⚡ Скорость на GPU CUDA:  {tok_s:.2f} tok/s ⚡")
    print("------------------------------------------------------------")
    print("✅ НАСТОЯЩИЕ ВЕСА В VRAM ИНФЕРЯТСЯ НА 100% УСПЕШНО.")

if __name__ == "__main__":
    run_real_7b_cuda_test()
