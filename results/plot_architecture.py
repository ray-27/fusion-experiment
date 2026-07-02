"""Draws where each connector attaches to the frozen SigLIP -> Qwen2 pipeline,
grounded directly in src/vlm.py's actual wiring (not illustrative guesswork):
  - mlp_concat / qformer: "prefix" path -> _merge_prefix() splices connector
    output tokens into the input embedding sequence, before layer 1.
  - cross_attention: forward hooks attached to llm.model.layers[i] for
    i in {3,7,11,15,19,23} (0-indexed) -> after decoder layers 4/8/12/16/20/24.
"""

from pathlib import Path

import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent
OUT_PATH = RESULTS_DIR / "architecture.png"

NUM_LLM_LAYERS = 24
INJECT_LAYERS_1INDEXED = [4, 8, 12, 16, 20, 24]  # from CrossAttentionConnector defaults

FROZEN_COLOR = "#c9d6e3"
TRAIN_COLOR = "#f4b183"
LLM_COLOR = "#d9d9d9"


def box(ax, xy, w, h, text, color, fontsize=9, edge="black"):
    b = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.2, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return b


def arrow(ax, p1, p2, color="black", style="-|>", lw=1.4, connectionstyle="arc3"):
    a = FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=12, linewidth=lw,
        color=color, connectionstyle=connectionstyle,
    )
    ax.add_patch(a)


def draw_prefix_panel(ax, title, connector_label, connector_detail, n_tokens):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box(ax, (0.3, 7.8), 2.6, 1.2, "SigLIP\nvision encoder\n(frozen)", FROZEN_COLOR)
    box(ax, (0.3, 5.6), 2.6, 1.2, "196 patch\nembeddings\n(768-dim)", "white")
    box(ax, (0.3, 3.2), 2.6, 1.6, connector_label + "\n(TRAINABLE)\n" + connector_detail, TRAIN_COLOR)
    box(ax, (0.3, 1.0), 2.6, 1.2, f"{n_tokens} tokens\n(896-dim)", "white")

    arrow(ax, (1.6, 7.8), (1.6, 6.8))
    arrow(ax, (1.6, 5.6), (1.6, 4.8))
    arrow(ax, (1.6, 3.2), (1.6, 2.2))

    box(ax, (3.6, 0.6), 1.8, 1.0, "<image>\ntoken slot\nreplaced", "white", fontsize=8)
    arrow(ax, (2.9, 1.6), (3.6, 1.1))

    seq_x = 5.7
    box(ax, (seq_x, 0.6), 3.9, 1.0,
        f"[text][{n_tokens}x image][text]\ninput sequence to Qwen2", "white", fontsize=8)
    arrow(ax, (5.4, 1.1), (seq_x, 1.1))

    llm_y = 2.2
    box(ax, (5.7, llm_y), 3.9, 5.6, "", LLM_COLOR)
    ax.text(7.65, llm_y + 5.3, "Qwen2-0.5B (frozen)\n24 decoder layers", ha="center", fontsize=9)
    for i in range(6):
        ly = llm_y + 0.3 + i * 0.85
        box(ax, (6.0, ly), 3.3, 0.6, f"layer {i * 4 + 1}-{i * 4 + 4}", LLM_COLOR, fontsize=7.5)
    arrow(ax, (7.65, 1.6), (7.65, llm_y))
    arrow(ax, (7.65, llm_y + 5.6), (7.65, 9.0))
    box(ax, (6.5, 9.0), 2.3, 0.7, "output logits", "white", fontsize=8)


def draw_cross_attn_panel(ax):
    ax.set_title("Mechanism #3: cross_attention\n(Flamingo-style)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box(ax, (0.3, 8.0), 2.4, 1.0, "SigLIP\nvision encoder\n(frozen)", FROZEN_COLOR, fontsize=8)
    box(ax, (0.3, 6.4), 2.4, 1.0, "196 patch\nembeddings (768d)", "white", fontsize=8)
    box(ax, (0.3, 4.8), 2.4, 1.0, "vision_proj\nLinear 768->896\n(TRAINABLE)", TRAIN_COLOR, fontsize=8)
    arrow(ax, (1.5, 8.0), (1.5, 7.4))
    arrow(ax, (1.5, 6.4), (1.5, 5.8))
    ax.text(1.5, 4.4, "cached image_features\n(fed to every gated block)", ha="center", fontsize=7.5, style="italic")

    box(ax, (0.3, 0.6), 2.4, 1.0, "text tokens\n(unchanged length)", "white", fontsize=8)
    arrow(ax, (1.5, 1.6), (1.5, 2.2))

    llm_x, llm_y, llm_w, llm_h = 3.4, 0.4, 2.6, 9.0
    box(ax, (llm_x, llm_y), llm_w, llm_h, "", LLM_COLOR)
    ax.text(llm_x + llm_w / 2, llm_y + llm_h - 0.35, "Qwen2-0.5B\n24 layers (frozen)",
            ha="center", fontsize=8.5)
    arrow(ax, (1.5, 1.6), (llm_x, 0.9))

    layer_h = (llm_h - 1.0) / 24
    gate_x = llm_x + llm_w + 0.6
    for i in range(24):
        ly = llm_y + 0.5 + i * layer_h
        is_inject = (i + 1) in INJECT_LAYERS_1INDEXED
        box(ax, (llm_x + 0.15, ly), llm_w - 0.3, layer_h * 0.8,
            f"L{i + 1}", LLM_COLOR if not is_inject else "#eaeaea", fontsize=6)
        if is_inject:
            gy = ly + layer_h * 0.4
            box(ax, (gate_x, gy - 0.28), 2.6, 0.56,
                "gated cross-attn\n(TRAINABLE, tanh-gate)", TRAIN_COLOR, fontsize=6.5)
            arrow(ax, (llm_x + llm_w, gy), (gate_x, gy), color="#b35900", lw=1.0)
            arrow(ax, (1.5, 4.8), (gate_x + 0.2, gy), color="#b35900", lw=0.6,
                  connectionstyle="arc3,rad=0.15")

    arrow(ax, (llm_x + llm_w / 2, llm_y + llm_h), (llm_x + llm_w / 2, llm_y + llm_h + 0.6))
    box(ax, (llm_x - 0.2, llm_y + llm_h + 0.6), llm_w + 0.4, 0.6, "output logits", "white", fontsize=8)

    ax.text(9.2, 0.15,
            "hooked layers: 4, 8, 12, 16, 20, 24\n(after every 4th decoder layer)",
            ha="right", fontsize=7.5, style="italic")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(19, 8))

    draw_prefix_panel(
        axes[0], "Mechanism #1: mlp_concat\n(LLaVA-style)",
        "2-layer MLP\nLinear-GELU-Linear", "768 -> 896 -> 896\nper-patch, independent",
        n_tokens=196,
    )
    draw_prefix_panel(
        axes[1], "Mechanism #2: qformer\n(BLIP-2-style)",
        "Q-Former\n32 learned queries", "2x (self-attn +\ncross-attn + FFN)",
        n_tokens=32,
    )
    draw_cross_attn_panel(axes[2])

    frozen_patch = mpatches.Patch(color=FROZEN_COLOR, label="Frozen (SigLIP)")
    llm_patch = mpatches.Patch(color=LLM_COLOR, label="Frozen (Qwen2 layers)")
    train_patch = mpatches.Patch(color=TRAIN_COLOR, label="Trainable connector")
    fig.legend(handles=[frozen_patch, llm_patch, train_patch], loc="lower center", ncol=3, fontsize=10)

    fig.suptitle(
        "Where each connector attaches: frozen SigLIP -> connector -> frozen Qwen2-0.5B",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_PATH, dpi=150)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
