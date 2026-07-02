from connectors.cross_attention import CrossAttentionConnector
from connectors.mlp_concat import MLPConcatConnector
from connectors.qformer import QFormerConnector

# Ordered: this is the order runs execute when no --connector flag is given.
CONNECTORS = {
    "mlp_concat": MLPConcatConnector,
    "qformer": QFormerConnector,
    "cross_attention": CrossAttentionConnector,
}


def build_connector(name: str, vision_dim: int, llm_dim: int, **kwargs):
    if name not in CONNECTORS:
        raise ValueError(f"unknown connector {name!r}, choices: {list(CONNECTORS)}")
    return CONNECTORS[name](vision_dim, llm_dim, **kwargs)
