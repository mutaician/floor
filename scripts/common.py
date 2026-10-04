"""Small shared helpers for the reproducible preflight commands."""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def setup():
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = json.loads((ROOT / "experiment.json").read_text())
    if os.environ.get("FLOOR_BUDGET_USD"):
        limit = float(os.environ["FLOOR_BUDGET_USD"])
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("FLOOR_BUDGET_USD must be a positive finite number.")
        config["budget"].update(ceiling=limit, stop_scheduling_at=limit * 0.9)
    return config


def state_dir():
    """Private runtime files belong on a persistent volume when hosted."""
    return Path(os.environ.get("FLOOR_STATE_DIR", str(ROOT / "results")))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def normalize(text):
    return " ".join(text.casefold().split())
