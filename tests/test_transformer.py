import torch

from llm_from_scratch.transformer import TransformerBlock


def test_transformer():
    batch_size = 2
    n_token = 3
    emb_dim = 768
    cfg = {
        "context_length": 1024,
        "emb_dim": emb_dim,
        "n_heads": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }
    transformer = TransformerBlock(cfg)

    x = torch.rand(batch_size, n_token, emb_dim)
    out = transformer(x)
    assert out.shape == x.shape
