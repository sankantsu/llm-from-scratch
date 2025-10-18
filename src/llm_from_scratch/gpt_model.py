import torch
from torch import nn

from llm_from_scratch.gpt_config import GPTConfig
from llm_from_scratch.layer_norm import LayerNorm
from llm_from_scratch.transformer import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

        # Embeddings
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Transformer blocks
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.current_pos = 0

        # Final norm and linear layer
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        batch_size, seq_len = x.shape
        tok_embeds = self.tok_emb(x)

        # Position embedding
        if use_cache:
            pos_start = self.current_pos
            pos_end = self.current_pos + seq_len
            pos_ids = torch.arange(pos_start, pos_end, device=x.device)
            self.current_pos = pos_end
        else:
            pos_ids = torch.arange(seq_len, device=x.device)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        x = self.final_norm(x)
        return self.out_head(x)

    def reset_cache(self):
        for blk in self.trf_blocks:
            blk.reset_cache()
        self.current_pos = 0
