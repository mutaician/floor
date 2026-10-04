"""Verify Tinker capabilities and refresh pricing without generating tokens."""

import importlib.metadata
import json
import os
import urllib.error
import urllib.request

from common import ROOT, now, setup, write_json


def main():
    config = setup()
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is absent from the environment and .env.")

    import tinker

    service = tinker.ServiceClient()
    capabilities = service.get_server_capabilities().model_dump(mode="json")
    names = [item["model_name"] for item in capabilities["supported_models"]]
    target = config["model"]["id"]
    if target not in names:
        raise SystemExit(f"Requested model {target} is not listed for this account.")
    model = next(item for item in capabilities["supported_models"] if item["model_name"] == target)
    if model.get("trainable") is False or model.get("sampleable") is False:
        raise SystemExit("Requested model is not both trainable and sampleable.")
    write_json(ROOT / "results/access.json", {
        "checked_at_utc": now(),
        "sdk_version": importlib.metadata.version("tinker"),
        "requested_model": target,
        "model_access_verified": True,
        "supported_models": names,
        "model_capabilities": model,
        "paid_operations": 0,
        "note": "Capabilities only; generation, balance and queue times remain untested.",
    })
    print(f"Authenticated capabilities: {target} available.", flush=True)
    url = "https://tinker-docs.thinkingmachines.ai/tinker/models.json"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Floor-preflight/0.1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            prices = json.load(response)
    except urllib.error.URLError as error:
        write_json(ROOT / "results/pricing_fetch.json", {
            "checked_at_utc": now(), "source": url, "status": "failed",
            "error_type": type(error).__name__, "http_status": getattr(error, "code", None),
            "next_action": "Verify current official pricing through browser access before paid work.",
        })
        print("Pricing endpoint unavailable to this client; browser verification required.")
    else:
        write_json(ROOT / "results/prices.json", {"checked_at_utc": now(), "catalog": prices})
        print("Pricing catalog saved.")


if __name__ == "__main__":
    main()
