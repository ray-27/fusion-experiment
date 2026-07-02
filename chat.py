"""Two modes:

1. Plain text chat with the raw Qwen2 backbone (no vision fusion):
       python chat.py

2. Step-by-step VLM trace: manually pushes ONE DocVQA sample through
   FusionVLM (vision encoder -> connector -> LLM) and prints the output of
   every stage. No training happens; the connector is untrained (random
   init) unless a matching checkpoint exists in checkpoints/.
       python chat.py --vlm-trace
       python chat.py --vlm-trace --connector qformer --sample-idx 5
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch  # noqa: E402

from config import CHECKPOINTS_DIR  # noqa: E402
from connectors import build_connector  # noqa: E402
from data import DocVQADataset  # noqa: E402
from device import get_device  # noqa: E402
from models import load_llm, load_vision_encoder  # noqa: E402
from vlm import FusionVLM  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--vlm-trace", action="store_true", help="run the step-by-step VLM demo instead of chat")
    p.add_argument("--connector", default="mlp_concat", choices=["mlp_concat", "qformer", "cross_attention"])
    p.add_argument("--sample-idx", type=int, default=0)
    return p.parse_args()


def stats(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)} dtype={t.dtype} mean={t.mean().item():.4f} std={t.std().item():.4f}"


def run_vlm_trace(args):
    device = get_device()
    print(f"device: {device}")

    print("\n[load] vision encoder + Qwen2 + connector...")
    image_processor, vision_model = load_vision_encoder()
    tokenizer, llm = load_llm()
    vision_dim = vision_model.config.vision_config.hidden_size
    llm_dim = llm.config.hidden_size
    connector = build_connector(
        args.connector, vision_dim, llm_dim, num_llm_layers=llm.config.num_hidden_layers
    ).to(device)

    ckpt_path = CHECKPOINTS_DIR / f"connector_{args.connector}.pt"
    if ckpt_path.exists():
        connector.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[load] loaded trained connector weights from {ckpt_path}")
    else:
        print(f"[load] no checkpoint at {ckpt_path} -- connector is randomly initialized")

    vlm = FusionVLM(vision_model, connector, llm, tokenizer, image_processor).to(device)
    vlm.eval()  # inference only, no training in this script

    dataset = DocVQADataset()
    rec = dataset[args.sample_idx]
    print(f"\n[sample #{args.sample_idx}]")
    print(f"  question: {rec['question']!r}")
    print(f"  reference answers: {rec['answers']}")

    with torch.no_grad():
        print("\n--- step 1: SigLIP vision encoder ---")
        pixel_values = image_processor(images=rec["image"], return_tensors="pt")["pixel_values"].to(device)
        print(f"  pixel_values: {stats(pixel_values)}")
        patch_features = vlm._vision_features(pixel_values)
        print(f"  patch_features (SigLIP output): {stats(patch_features)}")

        print(f"\n--- step 2: connector ({args.connector}) ---")
        if vlm.fusion_type == "cross_attention":
            vlm.connector.set_image(patch_features)
            img_feats = vlm.connector.image_features
            print(f"  projected image_features cached for cross-attn: {stats(img_feats)}")
        else:
            image_embeds = vlm.connector(patch_features)
            print(f"  connector output tokens: {stats(image_embeds)}")

        prompt = f"<image>\n{rec['question']}\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
        print(f"\n--- step 3: build LLM input ---")
        print(f"  prompt: {prompt!r}")
        print(f"  tokenized ids shape: {tuple(ids.shape)}")

        if vlm.fusion_type == "cross_attention":
            input_ids, attention_mask, _ = vlm._pad_text([ids])
            print(f"  input_ids (unchanged length, image injected via hooks): {tuple(input_ids.shape)}")
            print("\n--- step 4: Qwen2 forward (with gated cross-attn hooks firing) ---")
            logits = vlm.llm(input_ids=input_ids, attention_mask=attention_mask).logits
        else:
            inputs_embeds, attention_mask, _ = vlm._merge_prefix(pixel_values, [ids])
            print(f"  merged inputs_embeds (image tokens spliced in): {stats(inputs_embeds)}")
            print("\n--- step 4: Qwen2 forward ---")
            logits = vlm.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits

        print(f"  logits: {stats(logits)}")
        top5 = torch.topk(logits[0, -1], 5)
        top5_tokens = [tokenizer.decode([t]) for t in top5.indices.tolist()]
        print(f"  top-5 next-token predictions: {list(zip(top5_tokens, top5.values.tolist()))}")

        print("\n--- step 5: vlm.generate() (greedy decode) ---")
        prediction = vlm.generate(rec["image"], rec["question"], max_new_tokens=args.max_new_tokens)
        print(f"  generated text: {prediction!r}")
        print(f"  reference answers: {rec['answers']}")

        if vlm.fusion_type == "cross_attention":
            vlm.connector.clear_image()


def run_chat(args):
    device = get_device()
    print(f"device: {device}")
    print("loading Qwen2...")
    tokenizer, model = load_llm()
    print("ready. type 'exit' or 'quit' to stop, 'reset' to clear history.\n")

    history = []
    while True:
        try:
            user_input = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "reset":
            history = []
            print("(history cleared)\n")
            continue

        history.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        if args.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=args.temperature)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)

        reply_ids = output[0, inputs.input_ids.shape[1]:]
        reply = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
        print(f"qwen: {reply}\n")
        history.append({"role": "assistant", "content": reply})


def main():
    args = parse_args()
    if args.vlm_trace:
        run_vlm_trace(args)
    else:
        run_chat(args)


if __name__ == "__main__":
    main()
