import argparse

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

from config import LLM_MODEL_CHOICES, MODELS_DIR, VISION_MODEL_ID
from hf_auth import get_hf_token


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--llm", default="small", choices=[*LLM_MODEL_CHOICES, "all"],
        help=(
            "which LLM backbone(s) to download: 'small' = Qwen2-0.5B (default, "
            "fast, ungated), 'large' = gemma-2-2b (~2B params, gated -- needs "
            "HF_TOKEN in .env + accepting the license at "
            "huggingface.co/google/gemma-2-2b), 'all' = both"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    token = get_hf_token()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.llm == "all":
        llm_ids = list(dict.fromkeys(LLM_MODEL_CHOICES.values()))  # dedup, keep order
    else:
        llm_ids = [LLM_MODEL_CHOICES[args.llm]]

    for model_id in (VISION_MODEL_ID, *llm_ids):
        print(f"downloading {model_id}")
        try:
            path = snapshot_download(repo_id=model_id, cache_dir=MODELS_DIR, token=token)
        except (GatedRepoError, HfHubHTTPError) as e:
            raise SystemExit(
                f"failed to download {model_id} -- it's gated on Hugging Face.\n"
                f"1. Accept the license at https://huggingface.co/{model_id}\n"
                f"2. Make sure HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is set in .env\n"
                f"original error: {e}"
            )
        print(f"  -> {path}")
    print("done")


if __name__ == "__main__":
    main()
