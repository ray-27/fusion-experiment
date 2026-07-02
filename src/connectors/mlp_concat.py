import torch.nn as nn


class MLPConcatConnector(nn.Module):
    """Mechanism #1: LLaVA-style 2-layer MLP projector, one output token per
    vision patch, concatenated straight into the LLM's input sequence."""

    fusion_type = "prefix"

    def __init__(self, vision_dim: int, llm_dim: int, **kwargs):
        super().__init__()
        hidden_dim = kwargs.get("hidden_dim") or llm_dim
        self.net = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
        )

    def forward(self, patch_features):
        # patch_features: (batch, num_patches, vision_dim)
        # returns: (batch, num_patches, llm_dim) -- token count unchanged.
        return self.net(patch_features)
