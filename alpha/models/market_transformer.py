"""Model class for Skill 6.2 (Ledger).

Mirrors notebooks/market_transformer_return_timing.ipynb's
`CausalReturnTransformer` exactly — must stay in sync with that notebook
for checkpoint loading to work. Imported lazily so `torch` isn't a hard
dependency of the app.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalReturnTransformer(nn.Module):
    def __init__(self, block_size: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, block_size, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        h = self.input_proj(x.unsqueeze(-1)) + self.pos_embedding[:, : x.size(1)]
        h = self.encoder(h, mask=self.causal_mask[: x.size(1), : x.size(1)])
        return self.head(h[:, -1, :]).squeeze(-1)
