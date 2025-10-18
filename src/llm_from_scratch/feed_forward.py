import torch
from torch import nn

from llm_from_scratch.gpt_config import GPTConfig


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        a = 0.5
        b = torch.sqrt(torch.tensor(2.0 / torch.pi))
        c = 0.044715
        return a * x * (1 + torch.tanh(b * (x + c * torch.pow(x, 3))))


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)
