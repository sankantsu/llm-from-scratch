import tiktoken
import torch


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_simple(
    model,
    batch: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    use_cache: bool = True,
):
    tok_idx = batch
    for i in range(max_new_tokens):
        if use_cache and i > 0:
            with torch.no_grad():
                # Feed only the last token
                logits = model(tok_idx[:, [-1]], use_cache=True)
        else:
            tok_idx = tok_idx[:, -context_size:]
            with torch.no_grad():
                logits = model(tok_idx)

        # Take only the last output token for each batch
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_tok_idx = torch.argmax(probs, dim=-1, keepdim=True)
        tok_idx = torch.cat((tok_idx, next_tok_idx), dim=-1)
    return tok_idx
