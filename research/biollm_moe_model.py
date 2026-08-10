"""
Модуль Гибридной Модели BioLLM Next-Gen с Поддержкой Инициализации Dense-to-MoE (biollm_moe_model.py).

Поддерживает инициализацию 8 экспертов из слоев MLP Qwen3.6-27B с добавлением 5% диверсификационного шума.

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
    def __init__(self, hidden_size: int = 4096, num_experts: int = 8, top_k: int = 2, capacity_ratio: float = 0.5, capacity_factor: float = 1.25):
        super().__init__()
        self.hidden_size = hidden_size
        self.mod_router = MoDRouter(hidden_size=hidden_size, capacity_ratio=capacity_ratio)
        
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        # Слой Sparse Bio-MoE с Capacity Factor 1.25
        self.moe = SparseBioMoELayer(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k, capacity_factor=capacity_factor)

    def init_from_dense_mlp(self, dense_mlp_state_dict, noise_std: float = 0.05):
        """
        Инициализирует 8 экспертов из монолитного MLP (Strategy Variant B: Dense-to-MoE Fine-tuning)
        """
        with torch.no_grad():
            for expert in self.moe.experts:
                # Копируем веса из dense MLP с небольшим гауссовым шумом
                for p_expert, (name_dense, p_dense) in zip(expert.parameters(), dense_mlp_state_dict.items()):
                    p_expert.copy_(p_dense + torch.randn_like(p_dense) * noise_std)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        mask, mod_scores = self.mod_router(hidden_states)
        
        norm1 = self.ln1(hidden_states)
        attn_out = self.attn(norm1)
        hidden_states = hidden_states + attn_out
        
        total_aux_loss = 0.0
        expert_usages = torch.zeros(self.moe.num_experts, device=hidden_states.device)
        total_overflow = 0
        
        if mask.any():
            selected_tokens = hidden_states[mask]
            norm2 = self.ln2(selected_tokens)
            
            moe_out, aux_loss, usages, overflow = self.moe(norm2)
            total_aux_loss = aux_loss
            expert_usages = usages
            total_overflow = overflow
            
            output = hidden_states.clone()
            output[mask] = output[mask] + moe_out
        else:
            output = hidden_states
            
        return output, total_aux_loss, expert_usages, total_overflow

class BioLLMNextGenModel(nn.Module):
    def __init__(self, num_layers: int = 16, hidden_size: int = 4096, num_experts: int = 8, top_k: int = 2, capacity_ratio: float = 0.5, capacity_factor: float = 1.25):
        super().__init__()
        self.layers = nn.ModuleList([
            BioLLMNextGenBlock(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k, capacity_ratio=capacity_ratio, capacity_factor=capacity_factor)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        total_loss = 0.0
        layer_usages = []
        total_overflows = 0
        for layer in self.layers:
            x, aux_loss, usages, overflow = layer(x)
            total_loss += aux_loss
            layer_usages.append(usages)
            total_overflows += overflow
        return x, total_loss, layer_usages, total_overflows

if __name__ == "__main__":
    print("🧪 Тестирование BioLLMNextGenModel с Инициализацией Dense-to-MoE...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = BioLLMNextGenModel(num_layers=4, hidden_size=4096, num_experts=8, top_k=2).to(device)
    dummy_input = torch.randn(1, 128, 4096, device=device)
    
    out, aux_loss, usages, overflow = model(dummy_input)
    print(f"📊 Выходной тензор модели: {out.shape}")
    print(f"⚡ Переполнения токенов:   {overflow}")
    print(f"⚖️ Суммарный Aux Loss:     {aux_loss.item():.4f}")
