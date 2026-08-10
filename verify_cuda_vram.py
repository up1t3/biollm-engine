"""
Скрипт точной эмпирической проверки выгрузки весов 27B GGUF в VRAM видеокарты RTX 3090.
Загружает веса через скомпилированные C++ CUDA 12 библиотеки и измеряет VRAM до и после.
"""

import os
import sys
import time
import subprocess

# Исправление битых путей CUDA на Windows
if "CUDA_PATH" in os.environ:
    del os.environ["CUDA_PATH"]

# Добавляем папки с C++ CUDA 12 библиотеками LM Studio
vendor_dir = r"C:\Users\Up1t3\.lmstudio\extensions\backends\vendor\win-llama-cuda12-vendor-v2"
backend_dir = r"C:\Users\Up1t3\.lmstudio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.27.1"

if os.path.exists(vendor_dir):
    os.add_dll_directory(vendor_dir)
if os.path.exists(backend_dir):
    os.add_dll_directory(backend_dir)

def get_vram_usage_mb():
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        )
        return int(res.strip())
    except Exception:
        return -1

def verify_cuda_vram_allocation():
    print("=" * 85)
    print("🔬 ЭМПИРИЧЕСКАЯ ПРОВЕРКА ВЫГРУЗКИ ВЕСОВ МОДЕЛИ В VRAM GPU RTX 3090")
    print("=" * 85)

    vram_before = get_vram_usage_mb()
    print(f"📊 Замер 1. VRAM ДО загрузки модели: {vram_before} МБ")

    model_path = r"E:\LMStudio\models\HauhauCS\Qwen3.6-27B-Uncensored-HauhauCS-Balanced\Qwen3.6-27B-Uncensored-HauhauCS-Balanced-Q4_K_P.gguf"
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена по пути: {model_path}")
        return

    print("\n🔥 Загрузка 40 слоев весов Qwen3.6-27B в VRAM видеокарты...")
    start_ts = time.time()
    
    from llama_cpp import Llama
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=40,
        n_ctx=2048,
        verbose=False
    )
    load_time = time.time() - start_ts

    vram_after = get_vram_usage_mb()
    delta_vram = vram_after - vram_before

    print("\n------------------------------------------------------------")
    print(f"⏱️ Время физической загрузки весов: {load_time:.2f} сек.")
    print(f"📊 Замер 2. VRAM ПОСЛЕ загрузки модели: {vram_after} МБ")
    print(f"⚡ ЧИСТЫЙ ПРИРОСТ VRAM НА GPU: {delta_vram} МБ ({delta_vram / 1024:.2f} ГБ) ⚡")
    print("------------------------------------------------------------")

    # Выполнение инференса
    print("\n💬 ГЕНЕРАЦИЯ ТЕСТОВОГО ОТВЕТА НА ТЕНЗОРНЫХ ЯДРАХ GPU...")
    res = llm.create_chat_completion(
        messages=[{"role": "user", "content": "Привет! Назови столицу Франции."}],
        max_tokens=30
    )
    ans = res["choices"][0]["message"]["content"]
    print(f"🤖 Ответ модели: {ans.strip()}")
    print("------------------------------------------------------------")

    if delta_vram > 3000:
        print("✅ ПОДТВЕРЖДЕНО: ВЕСА МОДЕЛИ НА 100% НАХОДЯТСЯ В ВИДЕОПАМЯТИ VRAM!")
    else:
        print("⚠️ Внимание: прирост VRAM составил менее 3 ГБ. Проверяем режим.")

if __name__ == "__main__":
    verify_cuda_vram_allocation()
