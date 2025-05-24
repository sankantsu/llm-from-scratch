import torch

from llm_from_scratch.dataset import create_dataloader_v1, get_verdict_txt


def test_get_verdict_txt():
    data = get_verdict_txt()
    prefix = "I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no g"
    assert isinstance(data, str)
    assert len(data) == 20479
    assert data[: len(prefix)] == prefix


def test_dataloader():
    text = get_verdict_txt()
    batch_size = 8
    max_length = 4
    stride = max_length
    dataloader = create_dataloader_v1(
        text,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=False,
    )
    assert len(dataloader) == 160
    input_tokens, target = next(iter(dataloader))
    assert input_tokens.shape == torch.Size((batch_size, max_length))
    assert target.shape == torch.Size((batch_size, max_length))
