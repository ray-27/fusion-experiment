import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
)

from config import (
    BASELINE_MAX_PIXELS,
    BASELINE_MIN_PIXELS,
    BASELINE_VLM_MODEL_ID,
    LLM_MODEL_ID,
    MODELS_DIR,
    VISION_MODEL_ID,
)
from device import get_device
from hf_auth import get_hf_token


def _inference_dtype(device: torch.device) -> torch.dtype:
    # bf16 roughly halves model + activation memory and is faster on CUDA
    # Tensor Cores. Kept at fp32 on CPU/MPS where bf16 support/perf is patchy.
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def load_vision_encoder():
    token = get_hf_token()
    processor = AutoProcessor.from_pretrained(
        VISION_MODEL_ID, cache_dir=MODELS_DIR, token=token
    )
    model = AutoModel.from_pretrained(
        VISION_MODEL_ID, cache_dir=MODELS_DIR, token=token
    )
    model = model.to(get_device()).eval()
    return processor, model


def load_llm():
    token = get_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_ID, cache_dir=MODELS_DIR, token=token
    )
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID, cache_dir=MODELS_DIR, token=token, torch_dtype=torch.float32
    )
    model = model.to(get_device()).eval()
    return tokenizer, model


def load_baseline_vlm(min_pixels=BASELINE_MIN_PIXELS, max_pixels=BASELINE_MAX_PIXELS):
    """Natively-multimodal reference model (Qwen2-VL). Not part of the frozen-
    backbone fusion ablation -- used only as a real-world upper-bound comparison.

    min_pixels/max_pixels cap Qwen2-VL's dynamic image resolution -- without
    this, large document scans can blow vision-tower attention memory past
    20GB for a single image. See config.py for details."""
    token = get_hf_token()
    processor = AutoProcessor.from_pretrained(
        BASELINE_VLM_MODEL_ID,
        cache_dir=MODELS_DIR,
        token=token,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    processor.tokenizer.padding_side = "left"  # required for batched decoder-only generation
    device = get_device()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASELINE_VLM_MODEL_ID, cache_dir=MODELS_DIR, token=token, torch_dtype=_inference_dtype(device)
    )
    model = model.to(device).eval()
    return processor, model
