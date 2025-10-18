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

        # KV cache: (batch_size, num_tokens, num_heads, head_dim)
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)
        self.ptr_current_pos = 0

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        batch_size, num_tokens, d_in = x.shape
        if num_tokens > self.context_length:
            msg = (
                "Number of input tokens must be smaller than or equal to context length"
            )
            raise ValueError(msg)

        if use_cache and self.k_cache is not None:
            assert num_tokens == 1, "Only the last token should be fed."

        queries = self.W_query(x).view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )
        keys_new = self.W_key(x).view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )
        values_new = self.W_value(x).view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )

        if use_cache:
            if self.k_cache is None:
                assert self.v_cache is None
                # Prefill
                self.k_cache = keys_new
                self.v_cache = values_new
            else:
                # Decode
                # Append new key and values to cache
                assert keys_new.shape[1] == 1 and values_new.shape[1] == 1
                self.k_cache = torch.cat([self.k_cache, keys_new], dim=1)
                self.v_cache = torch.cat([self.v_cache, values_new], dim=1)
            keys, values = self.k_cache, self.v_cache
        else:
            keys, values = keys_new, values_new

        # (batch_size, num_tokens, num_heads, head_dim) -> (batch_size, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        atten_scores = queries @ keys.transpose(2, 3)

        # Causal mask
        if use_cache:
            num_tokens_new = queries.shape[2]
            num_tokens_total = keys.shape[2]
            # Prefill: num_tokens_new == num_tokens_total
            # Decode: num_tokens_new == 1
            assert num_tokens_new == num_tokens_total or num_tokens_new == 1
            pos_start = self.ptr_current_pos
            pos_end = self.ptr_current_pos + num_tokens_new
            mask = self.mask.bool()[pos_start:pos_end, :num_tokens_total]
            self.ptr_current_pos = pos_end
        else:
            mask = self.mask.bool()[:num_tokens, :num_tokens]  # type: ignore[operator]
        atten_scores.masked_fill_(
            mask,
            -torch.inf,
        )
        atten_weights = torch.softmax(atten_scores / keys.shape[-1] ** 0.5, dim=-1)
        atten_weights = self.dropout(atten_weights)

        context_vecs = atten_weights @ values
        # (batch_size, num_heads, num_tokens, head_dim) -> (batch_size, num_tokens, d_out)
        context_vecs = (
            context_vecs.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, self.d_out)
        )
        return self.out_proj(context_vecs)

    def reset_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None
        self.ptr_current_pos = 0
