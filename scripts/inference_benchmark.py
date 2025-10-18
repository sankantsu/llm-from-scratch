import argparse
import time
import tiktoken
import torch
from llm_from_scratch.generate_text import (
    generate_text_simple,
)
from llm_from_scratch.gpt_model import GPTModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cache", action="store_true", help="Use KV cache.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Copied main() from:
    # https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/03_kv-cache/gpt_ch04.py
    # SPDX-FileCopyrightText: 2023-2025 Sebastian Raschka
    # SPDX-License-Identifier: Apache-2.0
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    token_ids = generate_text_simple(
        model,
        encoded_tensor,
        max_new_tokens=200,
        context_size=GPT_CONFIG_124M["context_length"],
        use_cache=args.use_cache,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024**3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
