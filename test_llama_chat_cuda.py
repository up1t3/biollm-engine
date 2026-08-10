"""
Прямой тест chat_completion через настоящие веса Qwen3.6-27B на GPU CUDA (RTX 3090).
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

GGUF_MODEL_27B = r"E:\LMStudio\models\HauhauCS\Qwen3.6-27B-Uncensored-HauhauCS-Balanced\Qwen3.6-27B-Uncensored-HauhauCS-Balanced-Q4_K_P.gguf"

def test_chat_cuda():
    print("=" * 85)
    print("🚀 ТЕСТ ЖИВОГО ДИАЛОГА QWEN3.6-27B GGUF С ПОЛНОЙ ЗАГРУЗКОЙ В VRAM")
    print("=" * 85)

    llm = Llama(
        model_path=GGUF_MODEL_27B,
        n_gpu_layers=99, # Все 64 слоя весов загружаются НАПРЯМУЮ в VRAM RTX 3090!
        n_ctx=4096,
        verbose=False
    )
    print("✅ Все 64 слоя весов Qwen3.6-27B физически загружены в VRAM!")

    messages = [
        {"role": "user", "content": "как зовут жену Билла Гейтса?"}
    ]

    start_t = time.time()
    resp = llm.create_chat_completion(
        messages=messages,
        max_tokens=128,
        temperature=0.7
    )
    elapsed = time.time() - start_t

    ans = resp["choices"][0]["message"]["content"]
    tok_count = resp["usage"]["completion_tokens"]
    tok_s = tok_count / max(elapsed, 0.01)

    print("\n------------------------------------------------------------")
    print("🤖 Ответ настоящей модели Qwen3.6-27B (CUDA GGUF VRAM):")
    print("------------------------------------------------------------")
    print(ans.strip())
    print("------------------------------------------------------------")
    print(f"⏱️ Время генерации:       {elapsed:.2f} сек.")
    print(f"⚡ Скорость на GPU CUDA:  {tok_s:.2f} tok/s")
    print("------------------------------------------------------------")

if __name__ == "__main__":
    test_chat_cuda()
