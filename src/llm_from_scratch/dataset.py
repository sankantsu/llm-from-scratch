import urllib.request

import tiktoken
import torch
from tiktoken import Encoding
from torch.utils.data import DataLoader, Dataset


def get_verdict_txt() -> str:
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
    ) as f:
        data = f.read()
    return data.decode(encoding="utf-8")


class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer: Encoding, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    *,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
