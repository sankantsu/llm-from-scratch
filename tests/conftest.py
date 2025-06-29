import pytest
import tiktoken


@pytest.fixture
def tokenizer():
    return tiktoken.get_encoding("gpt2")
