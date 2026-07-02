import os
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def get_hf_token() -> str | None:
    """Return the HF token from .env, else from the system environment."""
    load_dotenv(_ENV_PATH)
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(key)
        if token:
            return token.strip()
    return None
