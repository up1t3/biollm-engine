@echo off
TITLE BioLLM Engine v6.0 - Production Server Monitor (Port 8000)
COLOR 0A

echo ===============================================================================
echo 🚀 СТАРТ PRODUCTION СЕРВЕРА BIOLLM ENGINE v6.0 (VS CODE REST API BRIDGE)
echo ===============================================================================
echo  • REST API Endpoint:   http://localhost:8000/v1
echo  • GPU Device:          NVIDIA RTX 3090 (24 GB VRAM)
echo  • Weight Storage:      E:\biollm_models\ (Base-4 2-bit DNA Quantized)
echo  • Context Cache:       Hymba Mamba-2 SSM (~50 MB for 1M tokens)
echo ===============================================================================
echo.

cd /d E:\
set PYTHONIOENCODING=utf-8
set OLLAMA_MODELS=E:\ollama_models

C:\Users\Up1t3\.gemini\antigravity\scratch\biollm\.venv\Scripts\python.exe C:\Users\Up1t3\.gemini\antigravity\scratch\biollm\biollm_vscode_bridge.py

pause
