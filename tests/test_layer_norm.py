import torch

from llm_from_scratch.layer_norm import LayerNorm


def test_layer_norm():
    batch_size = 2
    emb_dim = 768
    eps = 1e-3

    torch.manual_seed(123)
    batch = torch.randn(batch_size, emb_dim)
    ln = LayerNorm(emb_dim=emb_dim)
    out_ln = ln(batch)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, keepdim=True, correction=0)

    assert torch.all(mean.abs() < eps)
    assert torch.all((var - 1.0).abs() < eps)
