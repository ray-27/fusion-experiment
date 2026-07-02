import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from PIL import Image

from config import DATA_DIR
from connectors import build_connector
from device import get_device
from models import load_llm, load_vision_encoder
from vlm import FusionVLM


def load_one_sample():
    sample_json = DATA_DIR / "docvqa_sample" / "samples.json"
    records = json.loads(sample_json.read_text())
    rec = records[0]
    image = Image.open(sample_json.parent / rec["image"]).convert("RGB")
    return image, rec["question"], rec["answers"]


def main():
    device = get_device()
    print(f"device: {device}")

    image_processor, vision_model = load_vision_encoder()
    tokenizer, llm = load_llm()

    vision_dim = vision_model.config.vision_config.hidden_size
    llm_dim = llm.config.hidden_size
    connector = build_connector("mlp_concat", vision_dim, llm_dim).to(device)
    print(f"connector: mlp_concat  {vision_dim} -> {llm_dim}")

    vlm = FusionVLM(vision_model, connector, llm, tokenizer, image_processor).to(device)

    image, question, answers = load_one_sample()
    print(f"question: {question!r}")
    print(f"reference answers: {answers}")

    output = vlm.generate(image, question, max_new_tokens=24)
    print(f"generated (untrained connector, expect noise): {output!r}")
    print("fusion forward pass ok")


if __name__ == "__main__":
    main()
