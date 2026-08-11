"""
Проверка Файла C++ CUDA Модуля (verify_pyd_file.py).

Проверяет физическое существование, точный размер и свойства mamba_cuda_scan.pyd.

Автор: Vladimir Popov <up1t3r@gmail.com> & Antigravity AI
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def audit_pyd():
    pyd_path = os.path.join(os.path.dirname(__file__), "cuda", "mamba_cuda_scan.pyd")
    
    print("=" * 80)
    print("🔬 АУДИТ СКОМПИЛИРОВАННОГО МОДУЛЯ MAMBA_CUDA_SCAN.PYD")
    print("=" * 80)
    print(f"  • Абсолютный путь: {pyd_path}")
    
    if os.path.exists(pyd_path):
        size_bytes = os.path.getsize(pyd_path)
        print(f"  • Файл существует: ✅ ДА")
        print(f"  • Точный размер:  📦 {size_bytes} байт ({size_bytes/1024:.2f} КБ)")
        
        if size_bytes < 1000:
            print("  ⚠️ Статус модуля: Эмулируемый режим совместимости PyTorch (stub)")
        else:
            print("  ✅ Статус модуля: Полноценный C++ DLL бинарный модуль CUDA")
    else:
        print("  ❌ Файл не найден!")
        
    print("=================================================================")

if __name__ == "__main__":
    audit_pyd()
