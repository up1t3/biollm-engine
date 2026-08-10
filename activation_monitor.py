"""
Модуль телеметрии и мониторинга здоровья активаций (Activation Monitor / Cellular Stress Response).
Отслеживает всплески шума, аномалии и энтропию на каждом слое для доказательного срабатывания Proofreader.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List

class ActivationHealthMonitor:
    """
    Монитор здоровья активаций: собирает метрики max_abs, std, spike_ratio, nan_ratio.
    """
    def __init__(self, spike_threshold_sigma: float = 6.0):
        self.spike_threshold_sigma = spike_threshold_sigma
        self.layer_stats: List[Dict[str, Any]] = []

    def inspect_activation(self, layer_name: str, x: torch.Tensor) -> Dict[str, Any]:
        """
        Проводит медицинский осмотр активационного тензора х.
        """
        xf = x.float()
        
        is_finite = torch.isfinite(xf).all().item()
        max_abs = xf.abs().max().item()
        mean_val = xf.mean().item()
        std_val = xf.std(dim=-1).mean().item()

        # Порог обнаружения аномальных выбросов (spikes)
        sigma_clamp = max(std_val * self.spike_threshold_sigma, 1e-6)
        spikes_count = (xf.abs() > sigma_clamp).float().mean().item()

        stat = {
            "layer_name": layer_name,
            "is_finite": is_finite,
            "max_abs": max_abs,
            "mean": mean_val,
            "std": std_val,
            "spike_ratio": spikes_count
        }

        self.layer_stats.append(stat)
        return stat

    def print_health_report(self):
        """
        Печатает сводный отчет о здоровье всех слоев модели.
        """
        print("\n🏥 ОТЧЕТ О ЗДОРОВЬЕ АКТИВАЦИЙ МОДЕЛИ (Activation Health Report):")
        print("-" * 75)
        print(f"{'Слой':<30} | {'Finite':<8} | {'MaxAbs':<10} | {'Std':<10} | {'SpikeRatio':<10}")
        print("-" * 75)
        for s in self.layer_stats:
            finite_str = "OK" if s['is_finite'] else "FAIL"
            print(f"{s['layer_name']:<30} | {finite_str:<8} | {s['max_abs']:<10.4f} | {s['std']:<10.4f} | {s['spike_ratio']:<10.6f}")
        print("-" * 75)
