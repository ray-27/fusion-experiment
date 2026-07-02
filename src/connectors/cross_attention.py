import torch
import torch.nn as nn


class GatedCrossAttentionBlock(nn.Module):
    """Flamingo-style gated cross-attention. Gates start at 0 (tanh(0) == 0), so
    the block is an identity at init and the frozen LLM is undisturbed until the
    connector learns to open the gates."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.ff_gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, image_features):
        h = self.attn_norm(hidden_states)
        attn_out = self.attn(h, image_features, image_features)[0]
        hidden_states = hidden_states + attn_out * self.attn_gate.tanh()
        hidden_states = hidden_states + self.ff(self.ff_norm(hidden_states)) * self.ff_gate.tanh()
        return hidden_states


class CrossAttentionConnector(nn.Module):
    """Mechanism #3: gated cross-attention injected into the LLM's decoder
    layers. Instead of prepending image tokens, the text stream cross-attends to
    projected image patches at a subset of layers. FusionVLM wires the blocks in
    via forward hooks; this module owns the trainable weights and image cache."""

    fusion_type = "cross_attention"

    def __init__(self, vision_dim, llm_dim, **kwargs):
        super().__init__()
        num_llm_layers = kwargs["num_llm_layers"]
        inject_every = kwargs.get("inject_every", 4)
        num_heads = kwargs.get("num_heads", 8)

        self.vision_proj = nn.Linear(vision_dim, llm_dim)
        self.inject_layers = list(range(inject_every - 1, num_llm_layers, inject_every))
        self.blocks = nn.ModuleDict(
            {str(i): GatedCrossAttentionBlock(llm_dim, num_heads) for i in self.inject_layers}
        )
        self.image_features = None

    def set_image(self, patch_features):
        self.image_features = self.vision_proj(patch_features)

    def clear_image(self):
        self.image_features = None
