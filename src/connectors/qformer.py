import torch
import torch.nn as nn

from config import NUM_QUERY_TOKENS


class QFormerLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.sa_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ca_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, queries, context):
        h = self.sa_norm(queries)
        queries = queries + self.self_attn(h, h, h)[0]
        h = self.ca_norm(queries)
        queries = queries + self.cross_attn(h, context, context)[0]
        queries = queries + self.ff(self.ff_norm(queries))
        return queries


class QFormerConnector(nn.Module):
    """Mechanism #2: BLIP-2 style Q-Former. A fixed set of learned query tokens
    compresses the variable-length patch grid into `num_query_tokens` tokens via
    stacked self- + cross-attention, then feeds them into the LLM sequence."""

    fusion_type = "prefix"

    def __init__(self, vision_dim, llm_dim, **kwargs):
        super().__init__()
        num_query_tokens = kwargs.get("num_query_tokens", NUM_QUERY_TOKENS)
        num_heads = kwargs.get("num_heads", 8)
        num_layers = kwargs.get("num_layers", 2)

        self.queries = nn.Parameter(torch.randn(num_query_tokens, llm_dim) * 0.02)
        self.patch_proj = nn.Linear(vision_dim, llm_dim)
        self.layers = nn.ModuleList(
            [QFormerLayer(llm_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, patch_features):
        context = self.patch_proj(patch_features)
        queries = self.queries.unsqueeze(0).expand(patch_features.size(0), -1, -1)
        for layer in self.layers:
            queries = layer(queries, context)
        return queries  # (batch, num_query_tokens, llm_dim)
