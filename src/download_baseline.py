"""Downloads the natively-multimodal reference baseline (Qwen2-VL). This is
separate from download_models.py since it's not part of the controlled fusion
ablation -- it's a reference point trained end-to-end on massive multimodal
data, unlike SigLIP+Qwen2+connector which we train from scratch ourselves."""

from huggingface_hub import snapshot_download

from config import BASELINE_VLM_MODEL_ID, MODELS_DIR
from hf_auth import get_hf_token


def main():
    token = get_hf_token()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {BASELINE_VLM_MODEL_ID}")
    path = snapshot_download(repo_id=BASELINE_VLM_MODEL_ID, cache_dir=MODELS_DIR, token=token)
    print(f"  -> {path}")
    print("done")


if __name__ == "__main__":
    main()
