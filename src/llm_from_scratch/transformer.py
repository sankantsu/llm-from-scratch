import torch
from torch import nn

from llm_from_scratch.attention import MultiHeadAttention
from llm_from_scratch.feed_forward import FeedForward
from llm_from_scratch.layer_norm import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        context_length = cfg["context_length"]
        n_heads = cfg["n_heads"]
        dropout = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        self.attention = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=context_length,
            num_heads=n_heads,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )
        self.feed_forward = FeedForward(cfg)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # Self attention
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        # Feed forward
        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        return x + shortcut
