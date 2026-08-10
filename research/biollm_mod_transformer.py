"""
Модуль MoD Transformer Block & BioLLMMoDModel (biollm_mod_transformer.py).

Реализует слой Трансформера с выборочной обработкой сложных токенов через MoDRouter.

Автор: Vladimir Popov <up1t3r@gmail.com> & Antigravity AI
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))
from biollm_mod_router import MoDRouter

class MoDTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int = 4096, capacity_ratio: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.router = MoDRouter(hidden_size=hidden_size, capacity_ratio=capacity_ratio)
        
        # Стандартные слои внимания и MLP
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size, bias=False)
        )

    def forward(self, hidden_states: torch.Tensor):
        """
        hidden_states: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 1. Запрос к роутеру: какие токены обрабатывать?
        mask, scores = self.router(hidden_states) # [batch, seq_len]
        
        # 2. Формируем выходной тензор с инерционным Skip Connection
        output = hidden_states.clone()
        
        # 3. Применяем вычисления ТОЛЬКО к отбранным токенам
        if mask.any():
            selected_inputs = hidden_states[mask] # [num_selected, hidden_size]
            
            norm1 = self.ln1(selected_inputs)
            attn_out = self.attn(norm1)
            x_mid = selected_inputs + attn_out
            
            norm2 = self.ln2(x_mid)
            mlp_out = self.mlp(norm2)
            processed_out = x_mid + mlp_out
            
            # Вставляем обработанные значения обратно
            output[mask] = processed_out
            
        return output, mask

class BioLLMMoDModel(nn.Module):
    def __init__(self, num_layers: int = 64, hidden_size: int = 4096, capacity_ratio: float = 0.5):
        super().__init__()
        self.layers = nn.ModuleList([
            MoDTransformerBlock(hidden_size=hidden_size, capacity_ratio=capacity_ratio)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        masks = []
        for layer in self.layers:
            x, mask = layer(x)
            masks.append(mask)
        return x, masks

if __name__ == "__main__":
    print("🧪 Тестирование MoDTransformerBlock...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    block = MoDTransformerBlock(hidden_size=4096, capacity_ratio=0.5).to(device)
    dummy_input = torch.randn(1, 256, 4096, device=device)
    
    out, mask = block(dummy_input)
    print(f"📊 Выходной тензор: {out.shape}, Обработано токенов: {mask.sum().item()} из {mask.numel()}")
