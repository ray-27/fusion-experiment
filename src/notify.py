import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def get_discord_webhook() -> str | None:
    """.env first (DISCORD_WEBHOOK_URL), falling back to system env, same
    pattern as hf_auth.get_hf_token()."""
    load_dotenv(_ENV_PATH)
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    return url.strip() if url else None


def notify_discord(message: str) -> bool:
    """Best-effort webhook ping. Never raises -- a failed notification should
    never take down a training run."""
    url = get_discord_webhook()
    if not url:
        print("[notify] no DISCORD_WEBHOOK_URL in .env or system env -- skipping")
        return False

    payload = json.dumps({"content": message[:1900]}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = 200 <= resp.status < 300
        print(f"[notify] discord webhook sent (ok={ok})")
        return ok
    except urllib.error.URLError as e:
        print(f"[notify] failed to send discord webhook: {e}")
        return False
