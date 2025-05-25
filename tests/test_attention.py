import pytest
import torch
from llm_from_scratch.attention import MultiHeadAttention


def test_attention():
    d_in = 3
    d_out = 256
    num_tokens = 6

    dropout = 0.0
    context_length = num_tokens
    self_attention_layer = MultiHeadAttention(d_in, d_out, context_length=context_length, dropout=dropout, num_heads=2)

    batch_size = 8
    batch = torch.rand((batch_size, num_tokens, d_in))

    out = self_attention_layer(batch)
    assert out.shape == torch.Size((batch_size, num_tokens, d_out))


def test_attention_invalid_num_head():
    d_in = 3
    d_out = 256
    context_length = 8
    dropout = 0.0
    num_head = 3  # Not divisible
    with pytest.raises(ValueError) as exc:
        MultiHeadAttention(
            d_in, d_out, context_length=context_length, dropout=dropout, num_heads=num_head
        )
    assert "d_out must be divisible by num_heads" in str(exc.value)


def test_attention_num_tokens_exceed_context_length():
    d_in = 3
    d_out = 256
    context_length = 8
    dropout = 0.0
    num_head = 2  # Not divisible
    attention_layer = MultiHeadAttention(
        d_in, d_out, context_length=context_length, dropout=dropout, num_heads=num_head
    )

    batch_size = 1
    num_tokens = context_length + 1  # larger than context length
    batch = torch.rand((batch_size, num_tokens, d_in))
    with pytest.raises(ValueError) as exc:
        attention_layer(batch)
    assert "Number of input tokens must be smaller than or equal to context length" in str(exc.value)
