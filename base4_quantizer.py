"""
Модуль 2-битного четверичного (Base-4) квантования весов линейных слоев LLM.
Отображает тензоры FP16/BF16 в 4 дискретных био-состояния {-1.5, -0.5, +0.5, +1.5}.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional

class Base4Quantizer:
    """
    Класс квантователя весов в четверичную (Base-4) систему кодирования ДНК.
    Алфавит состояний:
        00 (A) -> -1.5
        01 (C) -> -0.5
        10 (G) -> +0.5
        11 (T) -> +1.5
    """

    ALPHABET = torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float32)

    def __init__(self, block_size: int = 128):
        """
        :param block_size: Размер блока для пер-канального / пер-блочного квантования.
        """
        self.block_size = block_size

    @staticmethod
    def quantize_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Квантует входной тензор FP16/FP32 в 2-битные индексы (0..3) и возвращает упакованный тензор с масштабами.
        
        :param tensor: Входной тензор весов [out_features, in_features]
        :return: (packed_quantized_weights, scale_factors, quantized_dequantized_float_weights)
        """
        original_shape = tensor.shape
        device = tensor.device
        dtype = tensor.dtype

        # 1. Вычисление пер-канального масштаба (Scale factor per row)
        # Scale = max(abs(W)) / 1.5
        max_vals = torch.max(torch.abs(tensor), dim=-1, keepdim=True).values
        scale = max_vals / 1.5
        scale = torch.clamp(scale, min=1e-8)  # Защита от деления на 0

        # 2. Нормализация весов
        scaled_tensor = tensor / scale

        # 3. Нахождение ближайшего значения из 4-ричного алфавита ДНК {-1.5, -0.5, 0.5, 1.5}
        # Алфавит: A=-1.5, C=-0.5, G=0.5, T=1.5
        alphabet = Base4Quantizer.ALPHABET.to(device)
        
        # Разница между значениями весов и 4 символами
        dists = torch.abs(scaled_tensor.unsqueeze(-1) - alphabet) # [out, in, 4]
        indices = torch.argmin(dists, dim=-1).to(torch.uint8)     # [out, in] со значениями 0, 1, 2, 3

        # 4. Восстановленный (деквантованный) тензор для проверки точности
        quantized_float = alphabet[indices.long()] * scale

        # 5. Упаковка 4 нуклеотидных индексов (по 2 бита) в 1 байт uint8
        # [00, 01, 10, 11] -> packing along last dim
        out_features, in_features = original_shape
        padded_in = (in_features + 3) // 4 * 4
        
        if padded_in != in_features:
            pad_amount = padded_in - in_features
            indices_padded = torch.nn.functional.pad(indices, (0, pad_amount), value=0)
        else:
            indices_padded = indices

        # Битовый сдвиг: b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)
        i0 = indices_padded[:, 0::4]
        i1 = indices_padded[:, 1::4] << 2
        i2 = indices_padded[:, 2::4] << 4
        i3 = indices_padded[:, 3::4] << 6
        
        packed_bytes = i0 | i1 | i2 | i3  # [out_features, in_features // 4] в формате uint8

        return packed_bytes, scale.to(dtype), quantized_float.to(dtype)

    @staticmethod
    def unpack_tensor(packed_bytes: torch.Tensor, scale: torch.Tensor, original_in_features: int) -> torch.Tensor:
        """
        Распаковывает uint8 байты обратно в деквантованные FP32/FP16 веса.
        
        :param packed_bytes: [out_features, packed_in_features] uint8
        :param scale: [out_features, 1]
        :param original_in_features: Исходная размерность входных фичей
        :return: [out_features, in_features] float тензор
        """
        device = packed_bytes.device
        alphabet = Base4Quantizer.ALPHABET.to(device)

        # Маскирование битов для извлечения 2-битных индексов
        b0 = packed_bytes & 0b00000011
        b1 = (packed_bytes >> 2) & 0b00000011
        b2 = (packed_bytes >> 4) & 0b00000011
        b3 = (packed_bytes >> 6) & 0b00000011

        # Объединение распакованных индексов
        unpacked_indices = torch.stack([b0, b1, b2, b3], dim=-1).reshape(packed_bytes.shape[0], -1)
        unpacked_indices = unpacked_indices[:, :original_in_features]

        # Декодирование из алфавита и умножение на scale
        unpacked_weights = alphabet[unpacked_indices.long()] * scale
        return unpacked_weights


def convert_linear_layer_to_base4(linear: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Вспомогательная функция для конвертации стандартного nn.Linear слоя в Base-4 формат.
    Возвращает упакованные веса, масштабы и среднеквадратичную ошибку (MSE).
    """
    with torch.no_grad():
        w_orig = linear.weight.data
        packed, scale, w_dequant = Base4Quantizer.quantize_tensor(w_orig)
        mse_error = torch.mean((w_orig - w_dequant) ** 2).item()
    return packed, scale, mse_error
