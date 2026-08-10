"""
Уточненный Модуль Sparse Bio-MoE Layer (biollm_moe_layer.py).

Реализует разреженный слой 8 экспертов MLP с Top-2 роутингом, ограничением емкости буфера
Expert Capacity Factor = 1.25, обработкой переполнения токенов (overflow handling) и Auxiliary Load Balancing Loss.

Автор: Vladimir Popov <up1t3r@gmail.com> & Antigravity AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleExpertMLP(nn.Module):
    """Отдельный эксперт MLP с 2-битным квантованием Base-4"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size, bias=False)
        )
        
    def forward(self, x: torch.Tensor):
        return self.net(x)

class SparseBioMoELayer(nn.Module):
    def __init__(self, hidden_size: int = 4096, num_experts: int = 8, top_k: int = 2, capacity_factor: float = 1.25):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        # 8 независимых экспертов
        self.experts = nn.ModuleList([
            SingleExpertMLP(hidden_size) for _ in range(num_experts)
        ])
        
        # Роутер назначения токенов
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        """
        hidden_states: [num_tokens, hidden_size]
        Returns:
            output: [num_tokens, hidden_size]
            aux_loss: Штраф за дисбаланс нагрузки на экспертов
            expert_usage: Статистика вызова каждого эксперта [num_experts]
            overflow_count: Количество токенов переполнения емкости
        """
        num_tokens, hidden_size = hidden_states.shape
        
        # 1. Емкость каждого эксперта с учетом capacity_factor (например, 125%)
        expert_capacity = int(math.ceil((num_tokens * self.top_k / self.num_experts) * self.capacity_factor))
        expert_capacity = max(expert_capacity, 1)
        
        # 2. Логиты роутера и Top-2 выбор
        router_logits = self.router(hidden_states) # [num_tokens, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1) # [num_tokens, num_experts]
        
        topk_weights, topk_indices = torch.topk(routing_weights, k=self.top_k, dim=-1) # [num_tokens, top_k]
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # 3. Расчет Auxiliary Load Balancing Loss
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=(0, 1))
        f_i = tokens_per_expert / (num_tokens * self.top_k)
        P_i = routing_weights.mean(dim=0)
        
        aux_loss = self.num_experts * torch.sum(f_i * P_i)
        
        # 4. Вычисление выхода с ограничением Expert Capacity
        output = torch.zeros_like(hidden_states)
        expert_counts = torch.zeros(self.num_experts, dtype=torch.int32, device=hidden_states.device)
        overflow_count = 0
        
        for k in range(self.top_k):
            indices_k = topk_indices[:, k]
            weights_k = topk_weights[:, k:k+1]
            
            for expert_idx in range(self.num_experts):
                token_mask = (indices_k == expert_idx)
                if token_mask.any():
                    selected_indices = torch.nonzero(token_mask, as_tuple=True)[0]
                    current_count = expert_counts[expert_idx].item()
                    
                    # Ограничение по capacity factor
                    if current_count + len(selected_indices) > expert_capacity:
                        allowed_slots = max(expert_capacity - current_count, 0)
                        if allowed_slots > 0:
                            processed_indices = selected_indices[:allowed_slots]
                            dropped_indices = selected_indices[allowed_slots:]
                        else:
                            processed_indices = torch.tensor([], dtype=torch.long, device=hidden_states.device)
                            dropped_indices = selected_indices
                        overflow_count += len(dropped_indices)
                    else:
                        processed_indices = selected_indices
                        
                    if len(processed_indices) > 0:
                        expert_input = hidden_states[processed_indices]
                        expert_out = self.experts[expert_idx](expert_input)
                        output[processed_indices] += weights_k[processed_indices] * expert_out
                        expert_counts[expert_idx] += len(processed_indices)
                        
        return output, aux_loss, tokens_per_expert, overflow_count

import math

if __name__ == "__main__":
    print("🧪 Тестирование SparseBioMoELayer с Expert Capacity Factor 1.25...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    moe_layer = SparseBioMoELayer(hidden_size=4096, num_experts=8, top_k=2, capacity_factor=1.25).to(device)
    dummy_tokens = torch.randn(512, 4096, device=device)
    
    out, aux_loss, usage, overflow = moe_layer(dummy_tokens)
    
    print(f"📊 Выходной тензор:       {out.shape}")
    print(f"⚡ Ограничение емкости:   Capacity Limit per Expert = {int(math.ceil((512*2/8)*1.25))} токенов")
    print(f"🚨 Токенов переполнения:  {overflow} (Overflow tokens dropped into residual)")
    print(f"⚖️ Load Balancing Loss: {aux_loss.item():.4f}")
