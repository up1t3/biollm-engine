"""
Модуль Гибридной Модели BioLLM Next-Gen (biollm_moe_model.py).

Объединяет Mixture-of-Depths (MoD 50% пропуск токенов) + Attention + Sparse Bio-MoE (Top-2 эксперта из 8).

Автор: Vladimir Popov <up1t3r@gmail.com> & Antigravity AI
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))
from biollm_mod_router import MoDRouter
from biollm_moe_layer import SparseBioMoELayer

class BioLLMNextGenBlock(nn.Module):
    def __init__(self, hidden_size: int = 4096, num_experts: int = 8, top_k: int = 2, capacity_ratio: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.mod_router = MoDRouter(hidden_size=hidden_size, capacity_ratio=capacity_ratio)
        
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        # Слой Sparse Bio-MoE заменяет стандартный монолитный MLP
        self.moe = SparseBioMoELayer(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k)

    def forward(self, hidden_states: torch.Tensor):
        """
        hidden_states: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 1. MoD Router: выбираем Top-50% сложных токенов
        mask, mod_scores = self.mod_router(hidden_states) # [batch, seq_len]
        
        # 2. Attention вычисляется для всех токенов
        norm1 = self.ln1(hidden_states)
        attn_out = self.attn(norm1)
        hidden_states = hidden_states + attn_out
        
        total_aux_loss = 0.0
        expert_usages = torch.zeros(self.moe.num_experts, device=hidden_states.device)
        
        # 3. Sparse MoE вычисляется ТОЛЬКО для отбранных сложных токенов
        if mask.any():
            selected_tokens = hidden_states[mask] # [num_selected, hidden_size]
            norm2 = self.ln2(selected_tokens)
            
            moe_out, aux_loss, usages = self.moe(norm2)
            total_aux_loss = aux_loss
            expert_usages = usages
            
            output = hidden_states.clone()
            output[mask] = output[mask] + moe_out
        else:
            output = hidden_states
            
        return output, total_aux_loss, expert_usages

class BioLLMNextGenModel(nn.Module):
    def __init__(self, num_layers: int = 16, hidden_size: int = 4096, num_experts: int = 8, top_k: int = 2, capacity_ratio: float = 0.5):
        super().__init__()
        self.layers = nn.ModuleList([
            BioLLMNextGenBlock(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k, capacity_ratio=capacity_ratio)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        total_loss = 0.0
        layer_usages = []
        for layer in self.layers:
            x, aux_loss, usages = layer(x)
            total_loss += aux_loss
            layer_usages.append(usages)
        return x, total_loss, layer_usages

if __name__ == "__main__":
    print("🧪 Тестирование BioLLMNextGenModel (MoD + Sparse Bio-MoE)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = BioLLMNextGenModel(num_layers=4, hidden_size=4096, num_experts=8, top_k=2).to(device)
    dummy_input = torch.randn(1, 128, 4096, device=device)
    
    out, aux_loss, usages = model(dummy_input)
    print(f"📊 Выходной тензор модели: {out.shape}")
    print(f"⚖️ Суммарный Aux Loss:     {aux_loss.item():.4f}")
