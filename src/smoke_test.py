import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image

from config import DATA_DIR
from device import get_device
from models import load_llm, load_vision_encoder


def test_vision():
    processor, model = load_vision_encoder()
    sample_json = DATA_DIR / "docvqa_sample" / "samples.json"
    if sample_json.exists():
        records = json.loads(sample_json.read_text())
        img_path = sample_json.parent / records[0]["image"]
        image = Image.open(img_path).convert("RGB")
    else:
        image = Image.new("RGB", (224, 224), color=(127, 127, 127))

    inputs = processor(images=image, return_tensors="pt").to(get_device())
    with torch.no_grad():
        out = model.vision_model(**inputs)
    feats = out.last_hidden_state
    print(f"[vision] patch features: {tuple(feats.shape)}")
    return feats.shape


def test_llm():
    tokenizer, model = load_llm()
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(get_device())
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"[llm] prompt: {prompt!r}")
    print(f"[llm] output: {text!r}")
    return text


def main():
    print(f"device: {get_device()}")
    test_vision()
    test_llm()
    print("smoke test passed")


if __name__ == "__main__":
    main()
