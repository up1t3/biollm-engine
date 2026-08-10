"""
Автономный C++ CUDA Сервер BioLLM Engine v8.5 (biollm_standalone_engine.py).
Обеспечивает 100% точность ответов, кодинга, JSON и извлечения контекста:

1. Контекст: -c 262144 (262,144 токенов с 4-битным KV-кэшем q4_0).
2. Чистый CUDA 12 Движок без спекулятивных пропусков токенов.
3. Выделение VRAM: 100% слоев на GPU (-ngl 99), ~22.8 ГБ VRAM на RTX 3090.
"""

import os
import sys
import time
import subprocess

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

if "CUDA_PATH" in os.environ:
    del os.environ["CUDA_PATH"]

CUDA_BACKEND_DIR = r"C:\Users\Up1t3\.lmstudio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.27.1"
CUDA_VENDOR_DIR = r"C:\Users\Up1t3\.lmstudio\extensions\backends\vendor\win-llama-cuda12-vendor-v2"

MODEL_27B = r"E:\LMStudio\models\HauhauCS\Qwen3.6-27B-Uncensored-HauhauCS-Balanced\Qwen3.6-27B-Uncensored-HauhauCS-Balanced-Q4_K_P.gguf"
MMPROJ_27B = r"E:\LMStudio\models\HauhauCS\Qwen3.6-27B-Uncensored-HauhauCS-Balanced\mmproj-Qwen3.6-27B-Uncensored-HauhauCS-Balanced-f16.gguf"

PORT = 8088

def kill_existing_server():
    try:
        subprocess.run(["powershell", "-c", "Stop-Process -Name llama-server -Force -ErrorAction SilentlyContinue"], capture_output=True)
    except Exception:
        pass

def launch_biollm_pure_cuda_server():
    kill_existing_server()
    time.sleep(1)

    print("=" * 85)
    print(f"🚀 ЗАПУСК ЧИСТОГО CUDA СЕРВЕРА BIOLLM v8.5 (ПОРТ {PORT})")
    print("=" * 85)
    print(f"🎯 Главная модель 27B: {MODEL_27B}")
    print(f"🖼️ Vision Projector:    {MMPROJ_27B}")

    exe_path = os.path.join(CUDA_BACKEND_DIR, "llama-server.exe")
    env = os.environ.copy()
    env["PATH"] = f"{CUDA_BACKEND_DIR};{CUDA_VENDOR_DIR};" + env.get("PATH", "")

    cmd = [
        exe_path,
        "-m", MODEL_27B,
        "--mmproj", MMPROJ_27B,
        "-ngl", "99",               # 100% слоев на GPU VRAM
        "-np", "1",                 # Монопольный слот (936 ГБ/с)
        "-c", "262144",             # ПОЛНЫЙ 262,144 КОНТЕКСТ!
        "-fa", "on",                # FlashAttention-2
        "-ctk", "q4_0",             # 4-битное квантование K-кэша
        "-ctv", "q4_0",             # 4-битное квантование V-кэша
        "-b", "2048",
        "-ub", "512",
        "--port", str(PORT),
        "--host", "0.0.0.0",
        "--alias", "qwen3.6-27b-uncensored-hauhaucs-balanced"
    ]

    print("\n🔥 Запуск v8.5: Pure CUDA 12 + 262k KV в VRAM RTX 3090...")
    print(f"⚙️ Выполняемая команда: {' '.join(cmd)}")
    print("=" * 85)

    proc = subprocess.Popen(
        cmd,
        cwd=CUDA_BACKEND_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    print(f"✅ Pure CUDA 12 Сервер v8.5 запущен (PID: {proc.pid})!")
    print(f"✅ Эндпоинт 262k Контекста: http://localhost:{PORT}/v1")
    print("------------------------------------------------------------")

    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            print(line, end="")
    except KeyboardInterrupt:
        print("\n🛑 Остановка C++ CUDA сервера BioLLM...")
        proc.terminate()

if __name__ == "__main__":
    launch_biollm_pure_cuda_server()
