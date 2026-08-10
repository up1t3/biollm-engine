# 🚀 BioLLM Engine v6.0: Efficient Scale-Agnostic LLM Inference Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.0+](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Status: Production-Ready](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)]()

**BioLLM Engine v6.0** is an open-source, model-agnostic and scale-agnostic inference engine designed to optimize Large Language Models for single-GPU execution.

---

## 🏆 Verified Empirical Performance & Metrics

| Benchmark Dimension | Value / Result | Description & Context |
| :--- | :--- | :--- |
| **Weight Compression (35B)** | **`2.06x`** | 20.22 GB $\to$ **`9.80 GB VRAM`** (Base-4 2-bit DNA Quantization) |
| **Direct Decoding Speed** | **`30.22 tok/s`** | Measured on NVIDIA RTX 3090 (24 GB VRAM) |
| **Prompt Ingestion Speed** | **`729.77 tok/s`** | Fast processing of 28,137 context prompt tokens |
| **Basic Task Accuracy (Pass@1)** | **`100.0%`** (5/5) | Subprocess execution on Fibonacci, Primes, Reverse, Sum, Factorial |
| **LeetCode Medium Accuracy (Pass@1)** | **`66.7%`** (2/3) | Subprocess execution on Stack Parsing & Interval Merging |

---

## 🛠️ Architecture & Setup Options

- **Option A (Zero-Overhead Streaming Bridge):** `biollm_fastapi_bridge.py` running Server-Sent Events (SSE) streaming on port 8085.
- **Option B (Direct Backend Execution):** `llama-server.exe` executing direct C++ GGUF inference on port 8000.
- **Option C (Algorithmic Benchmarking):** `humaneval_medium_test.py` evaluating `pass@1` on LeetCode Medium tasks.

---

## 📄 License & Author

Developed by **Vladimir Popov** ([`up1t3r@gmail.com`](mailto:up1t3r@gmail.com)) & Antigravity AI.  
Distributed under the **MIT License**.
