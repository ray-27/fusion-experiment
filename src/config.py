from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
CHECKPOINTS_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

# Fixed vision encoder for every run.
VISION_MODEL_ID = "google/siglip-base-patch16-224"

# LLM backbone (Phase 1, English).
LLM_MODEL_ID = "Qwen/Qwen2-0.5B"

# Optional larger ~2B-parameter text-only backbone, selected via `train.py
# --llm large`. Everything downstream (connector dims, cross-attention layer
# count, tokenizer/special-token handling) is already read off the loaded
# model/tokenizer dynamically, so no other code needs to change to swap it in.
# NOTE: gemma-2-2b is gated on Hugging Face -- you must accept the license at
# https://huggingface.co/google/gemma-2-2b and set HF_TOKEN in .env before
# using it (see hf_auth.py).
LLM_MODEL_ID_LARGE = "google/gemma-2-2b"

LLM_MODEL_CHOICES = {
    "small": LLM_MODEL_ID,        # Qwen2-0.5B -- default, fast, ungated
    "large": LLM_MODEL_ID_LARGE,  # gemma-2-2b -- ~2.6B params, gated (needs HF_TOKEN)
}

# Natively-multimodal reference baseline (NOT part of the controlled fusion
# ablation -- Qwen2-VL was end-to-end multimodally pretrained on massive data,
# unlike our from-scratch frozen-backbone connector training). Used only to see
# where our small-scale trained mechanisms land relative to a real production VLM.
BASELINE_VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# Qwen2-VL uses a dynamic-resolution vision tower: by default it will process
# an image at up to ~12.8M pixels, which blows up vision-tower attention
# memory (>20GB for a single document scan) on anything short of a big
# datacenter GPU. Cap it -- 1024*28*28 px (~896x896 equivalent) keeps document
# text legible while fitting comfortably on a Tesla T4/V100/A100.
BASELINE_MIN_PIXELS = 256 * 28 * 28
BASELINE_MAX_PIXELS = 1024 * 28 * 28

# Eval dataset -- final held-out test set (document VQA, has inline images).
# Uses the FIRST `SAMPLE_SIZE` examples of the official `validation` split.
# Never trained on.
DOCVQA_DATASET_ID = "lmms-lab/DocVQA"
DOCVQA_CONFIG = "DocVQA"
DOCVQA_SPLIT = "validation"
SAMPLE_SIZE = 100

# Training data: DocVQA's own official `train` split (in-domain document
# reading, not just generic captions -- directly matches the brief's target
# task). Not exposed via load_dataset(..., split="train") for this dataset's
# metadata, but the parquet shards are loadable directly (see
# download_train_data.py). Bump TRAIN_SAMPLE_SIZE up on the Tesla GPU.
TRAIN_DATASET_ID = "lmms-lab/DocVQA"
TRAIN_CONFIG = "DocVQA"
TRAIN_SAMPLE_SIZE = 2000

# Validation (per-epoch monitoring) -- a DIFFERENT slice of the official
# `validation` split than the test set above, so there's zero overlap.
# Skips past the first SAMPLE_SIZE validation examples (already used as the
# test set) before taking its own samples.
VAL_MONITOR_SAMPLE_SIZE = 50
VAL_MONITOR_SKIP = SAMPLE_SIZE

# Special token spliced into the prompt; its embedding slot gets replaced
# with projected image features before the LLM forward pass.
IMAGE_TOKEN = "<image>"

# Number of learned query tokens for the Q-Former connector (mechanism #2).
NUM_QUERY_TOKENS = 32
