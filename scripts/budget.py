"""Reserve paid work durably before dispatch; retain uncertain charges."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

from scripts.common import now, setup, state_dir


@contextmanager
def locked_ledger():
    path = state_dir() / "cost_ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            ledger = json.loads(path.read_text())
        else:
            budget = setup()["budget"]
            ledger = dict(ceiling=budget["ceiling"], stop_scheduling_at=budget["stop_scheduling_at"], requests=[])
        yield ledger
        ledger["actual_usd"] = sum(row.get("accounted_usd", 0) for row in ledger["requests"])
        ledger["committed_estimated_usd"] = sum(row["reserved_usd"] for row in ledger["requests"] if row["status"] in {"committed", "uncertain"})
        ledger["accounting_note"] = "Token-based charge estimate; uncached input rate used conservatively. Uncertain requests retain their reservations. Not a reconciled billing invoice."
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)


def sampling_rates(config):
    prices = json.loads((state_dir() / "prices.json").read_text())
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(prices["checked_at_utc"])).total_seconds()
    if age > 3600:
        raise RuntimeError("Refresh results/prices.json before scheduling more paid work.")
    row = next(item for item in prices["catalog"] if item["tinker_id"] == config["model"]["id"])
    return {"input": float(row["prefill"].removeprefix("$")), "output": float(row["sample"].removeprefix("$"))}


def reserve_training(config, request_id, processed_tokens, stage):
    sampling_rates(config)  # Enforce the same fresh-price requirement.
    prices = json.loads((state_dir() / "prices.json").read_text())
    price = next(item for item in prices["catalog"] if item["tinker_id"] == config["model"]["id"])
    rate = float(price["train"].removeprefix("$"))
    reserve_fixed(config, request_id, processed_tokens * rate / 1e6, stage,
                  {"processed_tokens": processed_tokens, "training_rate_per_million": rate})


def reserve_fixed(config, request_id, cost, stage, details=None):
    with locked_ledger() as ledger:
        if any(row["request_id"] == request_id for row in ledger["requests"]):
            raise RuntimeError(f"Operation {request_id} already recorded; never replay a training update.")
        committed = sum(row["reserved_usd"] for row in ledger["requests"] if row["status"] in {"committed", "uncertain"})
        accounted = sum(row.get("accounted_usd", 0) for row in ledger["requests"])
        limit = min(config["budget"]["ceiling"], config["budget"]["stop_scheduling_at"], ledger["ceiling"], ledger["stop_scheduling_at"])
        if accounted + committed + cost >= limit:
            raise RuntimeError("Budget stop threshold reached; no operation dispatched.")
        ledger["requests"].append(dict(request_id=request_id, stage=stage, status="committed",
            reserved_at_utc=now(), reserved_usd=cost, **(details or {})))


def finish_training(request_id, result):
    with locked_ledger() as ledger:
        row = next(row for row in ledger["requests"] if row["request_id"] == request_id)
        # Reserve the full unshifted sequence length, slightly conservatively.
        row.update(status="complete", completed_at_utc=now(), accounted_usd=row["reserved_usd"], result=result)


def reserve(config, request_id, input_tokens, max_output_tokens, stage):
    rates = sampling_rates(config)
    cost = (input_tokens * rates["input"] + max_output_tokens * rates["output"]) / 1e6
    with locked_ledger() as ledger:
        previous = next((row for row in ledger["requests"] if row["request_id"] == request_id), None)
        if previous:
            if previous["status"] == "complete":
                return previous["prediction"]
            raise RuntimeError(f"Request {request_id} is already committed or uncertain; reconcile it before retrying.")
        committed = sum(row["reserved_usd"] for row in ledger["requests"] if row["status"] in {"committed", "uncertain"})
        accounted = sum(row.get("accounted_usd", 0) for row in ledger["requests"])
        limit = min(config["budget"]["ceiling"], config["budget"]["stop_scheduling_at"], ledger["ceiling"], ledger["stop_scheduling_at"])
        if accounted + committed + cost >= limit:
            raise RuntimeError("Budget stop threshold reached; no request dispatched.")
        ledger["requests"].append({
            "request_id": request_id, "stage": stage, "status": "committed",
            "reserved_at_utc": now(), "reserved_usd": cost,
            "input_tokens": input_tokens, "max_output_tokens": max_output_tokens,
            "rates_per_million": rates,
        })
    return None


def finish(request_id, prediction):
    with locked_ledger() as ledger:
        row = next(row for row in ledger["requests"] if row["request_id"] == request_id)
        assert prediction["output_tokens"] <= row["max_output_tokens"]
        row["accounted_usd"] = (row["input_tokens"] * row["rates_per_million"]["input"] + prediction["output_tokens"] * row["rates_per_million"]["output"]) / 1e6
        row.update(status="complete", completed_at_utc=now(), prediction=prediction)


def uncertain(request_id, error):
    with locked_ledger() as ledger:
        row = next(row for row in ledger["requests"] if row["request_id"] == request_id)
        row.update(status="uncertain", error_type=type(error).__name__, failed_at_utc=now())
