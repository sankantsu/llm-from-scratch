import tiktoken
import torch

from llm_from_scratch.generate_text import (
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)
from llm_from_scratch.gpt_config import GPT_CONFIG_124M
from llm_from_scratch.gpt_model import GPTModel


def test_text_to_token_ids(tokenizer: tiktoken.Encoding):
    text = "Every effort moves you"
    tokens = text_to_token_ids(text, tokenizer)
    assert torch.all(tokens == torch.tensor([[6109, 3626, 6100, 345]]))


def test_token_ids_to_text(tokenizer: tiktoken.Encoding):
    token_ids = torch.tensor([[6109, 3626, 6100, 345]])
    text = token_ids_to_text(token_ids, tokenizer)
    assert text == "Every effort moves you"


def test_generate_text(tokenizer: tiktoken.Encoding):
    torch.manual_seed(123)

    initial_text = "Hello, I am"
    batch = text_to_token_ids(initial_text, tokenizer)

    max_new_tokens = 6
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    out = generate_text_simple(
        model,
        batch,
        max_new_tokens=max_new_tokens,
        context_size=GPT_CONFIG_124M["context_length"],
    )
    assert len(out.shape) == 2
    assert out.shape[0] == batch.shape[0]
    assert batch.shape[1] < out.shape[1]
    assert out.shape[1] <= batch.shape[1] + max_new_tokens

    generated_text = tokenizer.decode(out.squeeze(0).tolist())
    assert isinstance(generated_text, str)
