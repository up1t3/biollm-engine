@echo off
chcp 65001 > nul
title BioLLM Engine C++ CUDA 12 Server Monitor [Port 8088]
color 0A

echo =====================================================================================
echo 🚀 ЗАПУСК АВТОНОМНОГО C++ CUDA СЕРВЕРА BIOLLM ENGINE v8.0 С 262k КОНТЕКСТОМ
echo =====================================================================================

set "CUDA_BACKEND_DIR=C:\Users\Up1t3\.lmstudio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.27.1"
set "CUDA_VENDOR_DIR=C:\Users\Up1t3\.lmstudio\extensions\backends\vendor\win-llama-cuda12-vendor-v2"
set "MODEL_27B=E:\LMStudio\models\HauhauCS\Qwen3.6-27B-Uncensored-HauhauCS-Balanced\Qwen3.6-27B-Uncensored-HauhauCS-Balanced-Q4_K_P.gguf"
set "MMPROJ_27B=E:\LMStudio\models\HauhauCS\Qwen3.6-27B-Uncensored-HauhauCS-Balanced\mmproj-Qwen3.6-27B-Uncensored-HauhauCS-Balanced-f16.gguf"

set "PATH=%CUDA_BACKEND_DIR%;%CUDA_VENDOR_DIR%;%PATH%"

echo 📄 Модель: %MODEL_27B%
echo 🖼️ Vision:  %MMPROJ_27B%
echo 🧠 Контекст: 262,144 токенов
echo ⚡ Порт:     8088
echo =====================================================================================

"%CUDA_BACKEND_DIR%\llama-server.exe" -m "%MODEL_27B%" --mmproj "%MMPROJ_27B%" -ngl 99 -np 1 -c 262144 -fa on -ctk q4_0 -ctv q4_0 -b 2048 -ub 512 --port 8088 --host 0.0.0.0 --alias qwen3.6-27b-uncensored-hauhaucs-balanced

pause
