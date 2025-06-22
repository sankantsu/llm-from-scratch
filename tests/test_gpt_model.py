import pytest
import tiktoken
import torch

from llm_from_scratch.gpt_config import GPT_CONFIG_124M
from llm_from_scratch.gpt_model import GPTModel


@pytest.fixture
def tokenizer():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture
def batch(tokenizer: tiktoken.Encoding):
    texts = ["Every effort moves you", "Every day holds a"]
    tokens = [torch.tensor(tokenizer.encode(txt)) for txt in texts]
    return torch.stack(tokens, dim=0)


def test_gpt_model(batch: torch.Tensor):
    torch.manual_seed(123)

    config = GPT_CONFIG_124M
    model = GPTModel(config)

    batch_size, seq_len = batch.shape
    out = model(batch)
    assert out.shape == torch.Size((batch_size, seq_len, config["vocab_size"]))
