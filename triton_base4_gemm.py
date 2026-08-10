"""
Модуль высокопроизводительного умножения матриц Triton / PyTorch Base-4 GEMM.
Выполняет 2-битное умножение Y = X * W_Base4 с распаковкой упакованных байт uint8 на лету.
"""

import torch
import torch.nn as nn
from typing import Tuple

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Алфавит ДНК в формате PyTorch tensor
BASE4_ALPHABET = torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float32)

class Base4LinearFunction(torch.autograd.Function):
    """
    Кастомная autograd функция для 2-битного Base-4 линейного слоя.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, packed_weight: torch.Tensor, scale: torch.Tensor, in_features: int) -> torch.Tensor:
        """
        :param x: Входные активации [batch, in_features]
        :param packed_weight: Упакованные веса [out_features, in_features // 4] uint8
        :param scale: Пер-канальный масштаб [out_features, 1]
        :param in_features: Исходное число входов
        :return: Выходной тензор [batch, out_features]
        """
        device = x.device
        alphabet = BASE4_ALPHABET.to(device)

        # 1. Быстрая распаковка на PyTorch (Fallback implementation / Reference)
        b0 = packed_weight & 0b00000011
        b1 = (packed_weight >> 2) & 0b00000011
        b2 = (packed_weight >> 4) & 0b00000011
        b3 = (packed_weight >> 6) & 0b00000011

        unpacked_indices = torch.stack([b0, b1, b2, b3], dim=-1).reshape(packed_weight.shape[0], -1)
        unpacked_indices = unpacked_indices[:, :in_features]

        # 2. Декодирование весов из нуклеотидного алфавита {-1.5, -0.5, 0.5, 1.5}
        w_float = alphabet[unpacked_indices.long()] * scale # [out_features, in_features]

        # 3. Умножение матриц Y = X @ W^T
        output = torch.matmul(x, w_float.t())
        return output


class Base4Linear(nn.Module):
    """
    Линейный слой BioLLM, хранящий веса в 2-битном Base-4 упакованном формате uint8.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Размерность упакованных весов (4 нуклеотида на 1 байт uint8)
        packed_in_dim = (in_features + 3) // 4
        
        # Регистрируем тензоры весов и масштабов как параметры/буферы
        self.register_buffer("packed_weight", torch.zeros((out_features, packed_in_dim), dtype=torch.uint8))
        self.register_buffer("scale", torch.ones((out_features, 1), dtype=torch.float32))

    def load_from_float_linear(self, linear_layer: nn.Linear) -> float:
        """
        Загружает FP16/FP32 веса из стандартного nn.Linear слоя и квантует их в Base-4.
        """
        from base4_quantizer import Base4Quantizer
        
        with torch.no_grad():
            w_orig = linear_layer.weight.data
            packed, scale, w_dequant = Base4Quantizer.quantize_tensor(w_orig)
            
            self.packed_weight.copy_(packed)
            self.scale.copy_(scale)
            
            mse_error = torch.mean((w_orig - w_dequant) ** 2).item()
            return mse_error

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return Base4LinearFunction.apply(x, self.packed_weight, self.scale, self.in_features)
