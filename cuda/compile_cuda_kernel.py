"""
Компилятор C++ CUDA Расширения для PyTorch (compile_cuda_kernel.py).
Компилирует mamba_cuda_scan.cu в родной PyTorch C++ бинарный модуль.
"""

import os
import sys
import torch
from torch.utils.cpp_extension import load

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def build_cuda_kernel():
    cuda_dir = os.path.dirname(__file__)
    cuda_src = os.path.join(cuda_dir, "mamba_cuda_scan.cu")
    
    print("=" * 75)
    print("🚀 СТАРТ КОМПИЛЯЦИИ CUDA КЕРНЕЛА MAMBA_CUDA_SCAN.CU")
    print("=" * 75)
    print(f"  • CUDA src path: {cuda_src}")
    
    if not torch.cuda.is_available():
        print("  ⚠️ CUDA недоступна в текущем PyTorch окружении. Пропуск компиляции на хосте.")
        return False
        
    try:
        mamba_scan_cuda = load(
            name="mamba_cuda_scan",
            sources=[cuda_src],
            verbose=True
        )
        print("✅ CUDA кернел успешно скомпилирован в PyTorch C++ модуль!")
        return True
    except Exception as e:
        print(f"⚠️ Компиляция завершилась с ошибкой настройки: {e}")
        return False

if __name__ == "__main__":
    build_cuda_kernel()
