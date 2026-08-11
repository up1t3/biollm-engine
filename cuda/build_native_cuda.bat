@echo off
REM Скрипт Сборки Полноценного CUDA DLL Модуля mamba_cuda_scan.pyd через MSVC vcvars64.bat

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

echo ===========================================================================
echo 🚀 СТАРТ КОМПИЛЯЦИИ ПОЛНОЦЕННОГО CUDA БИНАРНИКА MAMBA_CUDA_SCAN.PYD
echo ===========================================================================

cd /d "C:\Users\Up1t3\.gemini\antigravity\scratch\biollm\cuda"

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe" -O3 --shared -Xcompiler "/O2 /MD" mamba_cuda_scan.cu -o mamba_cuda_scan.pyd -I"C:\Users\Up1t3\.gemini\antigravity\scratch\biollm\.venv\Include" -I"C:\Users\Up1t3\AppData\Local\Programs\Python\Python312\include"

if exist mamba_cuda_scan.pyd (
    echo ✅ CUDA БИНАРНЫЙ МОДУЛЬ УСПЕШНО СКОМПИЛИРОВАН!
) else (
    echo ⚠️ ИСПОЛЬЗУЕТСЯ ЭМУЛИРУЕМЫЙ МОДУЛЬ PYTORCH
)
