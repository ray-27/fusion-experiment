from huggingface_hub import snapshot_download

from config import LLM_MODEL_ID, MODELS_DIR, VISION_MODEL_ID
from hf_auth import get_hf_token


def main():
    token = get_hf_token()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for model_id in (VISION_MODEL_ID, LLM_MODEL_ID):
        print(f"downloading {model_id}")
        path = snapshot_download(
            repo_id=model_id, cache_dir=MODELS_DIR, token=token
        )
        print(f"  -> {path}")
    print("done")


if __name__ == "__main__":
    main()
