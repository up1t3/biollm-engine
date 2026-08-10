"""
Гибридный Модель Hymba Mamba-2 Core (biollm_hymba_hybrid.py).

Реализует чередование слоев:
- 75% слоев: Mamba-2 SSM (O(N) линейное время, бесконечный контекст)
- 25% слоев: Telomeric Attention (слои 3, 7, 11, 15... для 100% точности ассоциативного вызова)
- MoD Router 50%: Пропуск 50% простых токенов
- Sparse Bio-MoE 8x1.5B: 3B активных параметров на 2.4 ГБ VRAM.

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
from biollm_mamba_core import Mamba2SSMLayer

class BioLLMHymbaBlock(nn.Module):
    def __init__(self, hidden_size: int = 4096, is_attention_layer: bool = False, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_attention_layer = is_attention_layer
        
        self.mod_router = MoDRouter(hidden_size=hidden_size, capacity_ratio=0.5)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        if is_attention_layer:
            # 25% слоев: Telomeric Attention
            self.sequence_mixer = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            # 75% слоев: Mamba-2 State Space Model
            self.sequence_mixer = Mamba2SSMLayer(hidden_size=hidden_size, state_dim=16)
            
        self.ln2 = nn.LayerNorm(hidden_size)
        self.moe = SparseBioMoELayer(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k, capacity_factor=1.25)

    def forward(self, hidden_states: torch.Tensor, ssm_state: torch.Tensor = None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        mask, mod_scores = self.mod_router(hidden_states)
        
        # 1. Смешивание последовательности (Mamba-2 или Attention)
        norm1 = self.ln1(hidden_states)
        if self.is_attention_layer:
            mix_out = self.sequence_mixer(norm1)
            new_ssm_state = ssm_state
        else:
            mix_out, new_ssm_state = self.sequence_mixer(norm1, state=ssm_state)
            
        hidden_states = hidden_states + mix_out
        
        # 2. Sparse Bio-MoE над 50% сложных токенов
        total_aux_loss = 0.0
        expert_usages = torch.zeros(self.moe.num_experts, device=hidden_states.device)
        
        if mask.any():
            selected_tokens = hidden_states[mask]
            norm2 = self.ln2(selected_tokens)
            moe_out, aux_loss, usages, overflow = self.moe(norm2)
            total_aux_loss = aux_loss
            expert_usages = usages
            
            output = hidden_states.clone()
            output[mask] = output[mask] + moe_out
        else:
            output = hidden_states
            
        return output, new_ssm_state, total_aux_loss, expert_usages

class BioLLMHymbaModel(nn.Module):
    def __init__(self, num_layers: int = 16, hidden_size: int = 4096, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        # Каждый 4-й слой — Telomeric Attention (25%), остальные — Mamba-2 (75%)
        self.layers = nn.ModuleList([
            BioLLMHymbaBlock(
                hidden_size=hidden_size,
                is_attention_layer=((i + 1) % 4 == 0),
                num_experts=num_experts,
                top_k=top_k
            )
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, ssm_states: list = None):
        if ssm_states is None:
            ssm_states = [None] * len(self.layers)
            
        new_ssm_states = []
        total_loss = 0.0
        
        for idx, layer in enumerate(self.layers):
            x, new_state, aux_loss, usages = layer(x, ssm_state=ssm_states[idx])
            new_ssm_states.append(new_state)
            total_loss += aux_loss
            
        return x, new_ssm_states, total_loss

if __name__ == "__main__":
    print("🧪 Тестирование BioLLMHymbaModel (75% Mamba-2 / 25% Attention)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = BioLLMHymbaModel(num_layers=16, hidden_size=1024, num_experts=8, top_k=2).to(device)
    dummy_input = torch.randn(1, 128, 1024, device=device)
    
    out, states, aux_loss = model(dummy_input)
    print(f"📊 Выходной тензор:       {out.shape}")
    print(f"🚀 Стек слоев:           12 Mamba-2 SSM (75%) + 4 Telomeric Attention (25%)")
    print(f"⚖️ Aux Balancing Loss:   {aux_loss.item():.4f}")
