"""
Модуль Динамического Роутера Mixture-of-Depths (biollm_mod_router.py).

Осуществляет оценку сложности каждого токена последовательности и формирует бинарную маску:
- True: Токен сложный, направляется в глубокие слои Attention + MLP.
- False: Токен простой, направляется по прямому пути (Skip Connection / Identity).

Автор: Vladimir Popov <up1t3r@gmail.com> & Antigravity AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoDRouter(nn.Module):
    def __init__(self, hidden_size: int, capacity_ratio: float = 0.5):
        """
        hidden_size: Размерность скрытого состояния (например, 4096)
        capacity_ratio: Доля обрабатываемых токенов (0.5 = 50% самых сложных токенов)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.capacity_ratio = capacity_ratio

        # Легковесный MLP роутер оценки сложности токена
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states: torch.Tensor):
        """
        hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            mask: [batch_size, seq_len] (True = обрабатывать, False = пропустить)
            complexity_scores: [batch_size, seq_len, 1]
        """
        batch_size, seq_len, _ = hidden_states.shape
        complexity_scores = self.router(hidden_states).squeeze(-1) # [batch_size, seq_len]

        # Выбираем Top-K наисложнейших токенов по емкости capacity_ratio
        k = max(int(seq_len * self.capacity_ratio), 1)
        
        # Топ-K порог по батчу
        topk_vals, _ = torch.topk(complexity_scores, k=k, dim=-1)
        thresholds = topk_vals[:, -1].unsqueeze(-1) # [batch_size, 1]
        
        mask = complexity_scores >= thresholds # [batch_size, seq_len]
        return mask, complexity_scores

if __name__ == "__main__":
    print("🧪 Тестирование MoDRouter (Mixture-of-Depths Router)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    router = MoDRouter(hidden_size=4096, capacity_ratio=0.5).to(device)
    dummy_input = torch.randn(2, 512, 4096, device=device) # Batch 2, Seq 512, Hidden 4096
    
    mask, scores = router(dummy_input)
    selected_count = mask.sum().item()
    total_count = mask.numel()
    
    print(f"📊 Исходно токенов в батче:    {total_count}")
    print(f"🎯 Отобрано сложных токенов:   {selected_count} ({selected_count/total_count*100:.1f}%)")
    print(f"⚡ Средний сколл сложности:   {scores.mean().item():.4f}")
