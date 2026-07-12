"""Model classes for Skill 6.1 (Cortex).

Mirrors notebooks/node_transformer_sentiment_forecast.ipynb's architecture
exactly — this file must stay in sync with that notebook, since it's what
loads the `.pt` checkpoint the notebook exports. Only imported lazily
(inside prediction_stock.py) so `torch` isn't a hard dependency of the app.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """Per-node transformer over the lookback window."""

    def __init__(self, in_dim: int, d_model: int, n_heads: int, n_layers: int, lookback: int):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        h = self.input_proj(x) + self.pos_embedding[:, : x.size(1)]
        h = self.encoder(h)
        return h[:, -1, :]


class GraphAttentionLayer(nn.Module):
    """Cross-sectional attention across nodes, masked by the sector adjacency matrix."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, node_embeddings, adjacency):
        mask = (adjacency == 0)
        attn_out, _ = self.attn(node_embeddings, node_embeddings, node_embeddings, attn_mask=mask)
        return self.norm(node_embeddings + attn_out)


class SentimentFusion(nn.Module):
    """Encodes the sentiment window and gates it into the node embedding."""

    def __init__(self, d_model: int, lookback: int):
        super().__init__()
        self.sentiment_proj = nn.Sequential(
            nn.Linear(lookback, d_model), nn.ReLU(), nn.Linear(d_model, d_model),
        )
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, node_embedding, sentiment_window):
        s_embed = self.sentiment_proj(sentiment_window)
        gate = self.gate(torch.cat([node_embedding, s_embed], dim=-1))
        return node_embedding + gate * s_embed


class NodeTransformerForecaster(nn.Module):
    def __init__(self, n_nodes: int, in_dim: int, lookback: int, d_model: int, n_heads: int, n_temporal_layers: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.temporal_encoder = TemporalEncoder(in_dim, d_model, n_heads, n_temporal_layers, lookback)
        self.graph_attention = GraphAttentionLayer(d_model, n_heads)
        self.sentiment_fusion = SentimentFusion(d_model, lookback)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, sentiment, adjacency):
        B, L, N, Fdim = x.shape
        x_per_node = x.permute(0, 2, 1, 3).reshape(B * N, L, Fdim)
        node_embeddings = self.temporal_encoder(x_per_node).view(B, N, -1)
        node_embeddings = self.graph_attention(node_embeddings, adjacency)
        sentiment_per_node = sentiment.permute(0, 2, 1, 3).reshape(B * N, L)
        fused = self.sentiment_fusion(
            node_embeddings.reshape(B * N, -1), sentiment_per_node
        ).view(B, N, -1)
        return self.head(fused).squeeze(-1)
