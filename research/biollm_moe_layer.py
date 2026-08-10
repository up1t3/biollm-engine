"""
Модуль Sparse Bio-MoE Layer (biollm_moe_layer.py).

Реализует разреженный слой 8 экспертов MLP с Top-2 роутингом и расчетом Auxiliary Load Balancing Loss.

Автор: Vladimir Popov <up1t3r@gmail.com> & Antigravity AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleExpertMLP(nn.Module):
    """Отдельный эксперт MLP"""
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
    def __init__(self, hidden_size: int = 4096, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
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
        """
        num_tokens, hidden_size = hidden_states.shape
        
        # 1. Логиты роутера
        router_logits = self.router(hidden_states) # [num_tokens, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1) # [num_tokens, num_experts]
        
        # 2. Выбор Top-2 экспертов
        topk_weights, topk_indices = torch.topk(routing_weights, k=self.top_k, dim=-1) # [num_tokens, top_k]
        
        # Нормализация весов Top-2 экспертов
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # 3. Расчет Auxiliary Load Balancing Loss
        # f_i = доля токенов у каждого эксперта
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).float() # [num_tokens, top_k, num_experts]
        tokens_per_expert = expert_mask.sum(dim=(0, 1)) # [num_experts]
        f_i = tokens_per_expert / (num_tokens * self.top_k)
        
        # P_i = средняя вероятность вызова эксперта
        P_i = routing_weights.mean(dim=0) # [num_experts]
        
        aux_loss = self.num_experts * torch.sum(f_i * P_i)
        
        # 4. Вычисление выхода экспертов
        output = torch.zeros_like(hidden_states)
        
        for k in range(self.top_k):
            indices_k = topk_indices[:, k] # [num_tokens]
            weights_k = topk_weights[:, k:k+1] # [num_tokens, 1]
            
            for expert_idx in range(self.num_experts):
                token_mask = (indices_k == expert_idx)
                if token_mask.any():
                    expert_input = hidden_states[token_mask]
                    expert_out = self.experts[expert_idx](expert_input)
                    output[token_mask] += weights_k[token_mask] * expert_out
                    
        return output, aux_loss, tokens_per_expert

if __name__ == "__main__":
    print("🧪 Тестирование SparseBioMoELayer (8 экспертов, Top-2)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    moe_layer = SparseBioMoELayer(hidden_size=4096, num_experts=8, top_k=2).to(device)
    dummy_tokens = torch.randn(512, 4096, device=device) # 512 токенов
    
    out, aux_loss, usage = moe_layer(dummy_tokens)
    
    print(f"📊 Выходной тензор:       {out.shape}")
    print(f"⚖️ Load Balancing Loss: {aux_loss.item():.4f}")
    print("📈 Распределение вызовов экспертов (E1..E8):")
    for idx, count in enumerate(usage.tolist()):
        print(f"  • Эксперт {idx+1}: {int(count)} токенов ({count/(512*2)*100:.1f}%)")
