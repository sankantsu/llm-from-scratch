import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        context_length: int = 256,
        dropout: float = 0.1,
        num_heads: int = 12,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_out % num_heads != 0:
            msg = "d_out must be divisible by num_heads"
            raise ValueError(msg)

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.context_length = context_length

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x: torch.Tensor):
        batch_size, num_tokens, d_in = x.shape
        if num_tokens > self.context_length:
            msg = "Number of input tokens must be smaller than or equal to context length"
            raise ValueError(msg)

        # queries, keys, values: (batch_size, num_heads, num_tokens, head_dim)
        queries = self.W_query(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.W_key(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        atten_scores = queries @ keys.transpose(2, 3)
        atten_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        atten_weights = torch.softmax(atten_scores / keys.shape[-1] ** 0.5, dim=-1)
        atten_weights = self.dropout(atten_weights)

        context_vecs = atten_weights @ values
        # (batch_size, num_heads, num_tokens, head_dim) -> (batch_size, num_tokens, d_out)
        context_vecs = context_vecs.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        return self.out_proj(context_vecs)
