import torch


def generate_text_simple(model, batch: torch.Tensor, max_new_tokens: int, context_size: int):
    tok_idx = batch
    for _ in range(max_new_tokens):
        tok_idx = tok_idx[:, -context_size:]
        with torch.no_grad():
            logits = model(tok_idx)

        # Take only the last output token for each batch
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_tok_idx = torch.argmax(probs, dim=-1, keepdim=True)
        tok_idx = torch.cat((tok_idx, next_tok_idx), dim=-1)
    return tok_idx
