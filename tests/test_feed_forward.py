import torch

from llm_from_scratch.feed_forward import FeedForward


def test_feed_forward():
    batch_size = 2
    n_token = 3
    emb_dim = 768

    cfg = {"emb_dim": emb_dim}
    ffn = FeedForward(cfg)

    x = torch.rand(batch_size, n_token, emb_dim)
    out = ffn(x)
    assert out.shape == torch.Size((batch_size, n_token, emb_dim))
