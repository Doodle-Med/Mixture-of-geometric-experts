#!/usr/bin/env python3
"""Minimal concept-aware MGM demo."""
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Tuple

try:
    from nuanced_routing_v2 import NuancedGeometricRouter
except ImportError:
    class NuancedGeometricRouter(nn.Module):
        def __init__(self, input_dim: int, num_experts: int, k: int = 2):
            super().__init__()
            self.detector = nn.Linear(input_dim, 1)
            self.router = nn.Linear(input_dim, num_experts)
            self.k = k

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
            soph = torch.sigmoid(self.detector(x))
            logits = self.router(x)
            top, idx = torch.topk(logits * soph, self.k, dim=-1)
            weights = torch.zeros_like(logits)
            weights.scatter_(-1, idx, F.softmax(top, dim=-1))
            bal = F.mse_loss(weights.sum(0), torch.full_like(weights.sum(0), x.size(0) * self.k / weights.size(1)))
            return weights, bal, {"routing_weights": weights, "avg_sophistication": soph.mean().item()}

class ConceptAwareMGM(nn.Module):
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 256, num_experts: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position = nn.Parameter(torch.randn(512, hidden_dim))
        self.experts = nn.ModuleList([self._create_expert(hidden_dim) for _ in range(num_experts)])
        self.router = NuancedGeometricRouter(hidden_dim, num_experts, k=2)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

    def _create_expert(self, dim: int) -> nn.Module:
        return nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim), nn.Dropout(0.1))

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, seq = input_ids.shape
        hidden = self.embedding(input_ids) + self.position[:seq].unsqueeze(0)
        flat = hidden.view(-1, self.hidden_dim)
        weights, bal_loss, analysis = self.router(flat)
        outs = [expert(flat) for expert in self.experts]
        stacked = torch.stack(outs, 1)
        combined = (stacked * weights.unsqueeze(-1)).sum(1)
        logits = self.proj(self.norm(combined)).view(bsz, seq, -1)
        return {"logits": logits, "balance_loss": bal_loss, "routing_analysis": analysis}

    def generate_with_concept_analysis(self, prompt_ids: torch.Tensor, max_new_tokens: int = 10) -> Dict:
        self.eval()
        gen = prompt_ids.clone()
        journey = []
        with torch.no_grad():
            for step in range(max_new_tokens):
                out = self(gen)
                next_logits = out["logits"][:, -1, :]
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
                journey.append({"step": step, "token_id": next_id.item(), "sophistication": out["routing_analysis"].get("avg_sophistication", 0.0)})
                gen = torch.cat([gen, next_id], 1)
                if next_id.item() == 0:
                    break
        return {"generated_ids": gen, "concept_journey": journey}

if __name__ == "__main__":
    prompt = torch.randint(0, 100, (1, 5))
    model = ConceptAwareMGM()
    result = model.generate_with_concept_analysis(prompt)
    print(result)
