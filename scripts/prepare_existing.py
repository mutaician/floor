"""Download and prepare a small, dialogue-disjoint slice of CraigslistBargains.

Run with ``.venv/bin/python scripts/prepare_existing.py``. Source utterances are
kept verbatim; offer and seller-policy labels are derived for the Floor task.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, sha256, write_json, write_jsonl


DATASET = "stanfordnlp/craigslist_bargains"
REVISION = "cfb6992c5ca9bad209323ed8e42e0cfc7e4178cf"
RAW = ROOT / "data" / "raw_bargains"
URLS = {
    "train": "https://worksheets.codalab.org/rest/bundles/0xd34bbbc5fb3b4fccbd19e10756ca8dd7/contents/blob/parsed.json",
    "validation": "https://worksheets.codalab.org/rest/bundles/0x15c4160b43d44ee3a8386cca98da138c/contents/blob/parsed.json",
}
AMOUNT = re.compile(r"(?<![\w])\$\s*(\d[\d,]*(?:\.\d{1,2})?)(?![\w])")
AMBIGUOUS = re.compile(
    r"\b(?:off\s+(?:the\s+)?(?:listing\s+)?price|deposit|down\s+payment|"
    r"installments?|monthly|per\s+(?:month|week|day|hour)|shipping|delivery|"
    r"deliver(?:y|ed|ing)?|plus\s+(?:shipping|delivery)|including\s+shipping)\b",
    re.I,
)


def get_raw(split: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{split}.json"
    if not path.exists():
        with urlopen(URLS[split], timeout=90) as response:
            path.write_bytes(response.read())
    return path


def seller_and_item(scenario):
    kbs = scenario.get("kbs", [])
    sellers = [kb for kb in kbs if (kb.get("personal") or {}).get("Role") == "seller"]
    if len(sellers) != 1:
        return None
    return sellers[0]


def candidates(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    for dialogue in rows:
        scenario = dialogue.get("scenario") or {}
        seller = seller_and_item(scenario)
        if seller is None:
            continue
        item = seller.get("item") or {}
        asking_raw = item.get("Price")
        if not isinstance(asking_raw, (int, float)) or asking_raw <= 0 or not float(asking_raw).is_integer():
            continue
        asking = int(asking_raw)
        seller_target = (seller.get("personal") or {}).get("Target")
        if not isinstance(seller_target, (int, float)) or seller_target <= 0:
            seller_target = asking
        # Scenario order defines the agent IDs. Accept only a buyer's annotated
        # opening offer; its metadata price must equal the sole explicit dollar sum.
        seller_index = next(i for i, kb in enumerate(scenario["kbs"]) if kb is seller)
        events = dialogue.get("events", [])
        for event in events:
            if event.get("action") != "message" or event.get("agent") in (None, seller_index):
                continue
            msg = event.get("data")
            meta = event.get("metadata") or {}
            if not isinstance(msg, str) or not msg.strip() or meta.get("intent") != "init-price":
                continue
            if AMBIGUOUS.search(msg):
                continue
            amounts = AMOUNT.findall(msg)
            if len(amounts) != 1:
                continue
            offered = float(amounts[0].replace(",", ""))
            annotated = meta.get("price")
            if not offered.is_integer() or not isinstance(annotated, (int, float)) or float(annotated) != offered:
                continue
            if not 0 < offered <= asking:
                continue
            # init-price is the source annotation for a proposal, while requiring
            # a proposal verb/phrase avoids question-only price inquiries.
            if not re.search(r"\b(?:offer|pay|buy|purchase|take|sell|accept|do|go)\b", msg, re.I):
                continue
            # Exclude explicit side deals or non-cash consideration.
            if re.search(r"\b(?:trade|swap|throw\s+in|include\s+(?:my|a|an))\b", msg, re.I):
                continue
            dialogue_id = str(scenario.get("uuid") or dialogue.get("uuid") or "")
            if not dialogue_id:
                continue
            yield {
                "dialogue_id": dialogue_id,
                "item": item.get("Title") or item.get("Category") or "Craigslist item",
                "asking": asking,
                "seller_target": int(seller_target),
                "buyer_message": msg,
                "expected_offer": int(offered),
                "source_split": path.stem,
            }


def stable_key(row):
    return hashlib.sha256(("floor-existing-v1:" + row["dialogue_id"]).encode()).hexdigest()


def prepare():
    train_path = get_raw("train")
    # A full, fast pass over the source train split gives enough unique dialogues
    # to choose a disjoint 120/30 split without relying on pre-existing task labels.
    pool_by_id = {}
    for row in candidates(train_path):
        pool_by_id.setdefault(row["dialogue_id"], row)
    pool = sorted(pool_by_id.values(), key=stable_key)
    if len(pool) < 100:
        raise RuntimeError(f"Only {len(pool)} qualifying unique dialogues; refusing a weak slice")
    selected = pool[:150]
    holdout_ids = {row["dialogue_id"] for row in selected[:30]}
    output = []
    for row in selected:
        # The source contains no seller Bottomline values for this scenario set.
        # Use a deterministic synthetic task policy, clearly disclosed per row.
        minimum = min(row["asking"], max(1, math.ceil(row["seller_target"] * 0.7)))
        output.append({
            "id": "existing-" + row["dialogue_id"],
            "split": "validation" if row["dialogue_id"] in holdout_ids else "train",
            "family": "existing_dialogue",
            "origin": "human_craigslist_bargains",
            "dialogue_id": row["dialogue_id"],
            "item": row["item"],
            "currency": "USD",
            "asking": row["asking"],
            "minimum": minimum,
            "buyer_message": row["buyer_message"],
            "expected_offer": row["expected_offer"],
            "expected_payment": "full",
            "expected_seller_cost": 0,
            "expected_intent": "offer",
            "seller_policy_synthetic": True,
            "context": "Assume local pickup and full payment unless buyer says otherwise",
            "label_notes": "Offer is derived from the source init-price annotation and matching explicit dollar amount; minimum is a synthetic seller policy (ceil(70% of seller Target), capped at asking), not an observed source floor. Full payment and zero seller cost are task assumptions.",
        })
    write_jsonl(ROOT / "data" / "existing_bargains.jsonl", output)
    write_json(ROOT / "data" / "existing_source.json", {
        "dataset": DATASET,
        "revision": REVISION,
        "homepage": "https://stanfordnlp.github.io/cocoa/",
        "repository": "https://github.com/stanfordnlp/cocoa",
        "paper": "He, Chen, Balakrishnan, and Liang (2018), Decoupling Strategy and Generation in Negotiation Dialogues, arXiv:1808.09637",
        "citation": "@misc{he2018decoupling, title={Decoupling Strategy and Generation in Negotiation Dialogues}, author={He He and Derek Chen and Anusha Balakrishnan and Percy Liang}, year={2018}, eprint={1808.09637}, archivePrefix={arXiv}, primaryClass={cs.CL}}",
        "download_urls": URLS,
        "raw_files": {s: {"path": str((RAW / f'{s}.json').relative_to(ROOT)), "sha256": sha256(RAW / f"{s}.json")} for s in URLS},
        "source_splits_used": ["train"],
        "source_schema": "scenario.kbs[].personal (Role/Target/Bottomline), scenario.kbs[].item (Title/Category/Price), events[].agent/action/data/metadata (intent/price)",
        "source_dialogues_examined": 5247,
        "selection": "One buyer opening init-price message per unique dialogue; require exactly one explicit dollar amount, equal to source annotation price, with an offer/proposal verb and no detected payment, delivery, or side-deal ambiguity. Buyer text is copied verbatim. Deterministically selected 150 unique dialogues by SHA-256 of dialogue ID.",
        "task_labels": "Offer is derived from original machine-generated init-price metadata corroborated by message text. Listing ask comes from seller-side item Price. Seller minimum, full-payment assumption, and zero seller cost are synthetic task labels, not original ground truth. Minimum is min(asking, ceil(0.70 * seller Target)); if Target is absent, asking is its fallback. Local pickup is assumed, and delivery is excluded.",
        "split_policy": "30 separate dialogues reserved as validation; 120 train. Split IDs are disjoint. This is a derived split over source train dialogues, not the source validation split.",
        "count": len(output),
        "counts_by_split": {s: sum(r["split"] == s for r in output) for s in ("train", "validation")},
        "attribution": "Original human-human Craigslist negotiation dialogues collected by the Stanford NLP Cocoa project; source annotations were rules-based. Cite the paper above. Dataset card marks license unknown.",
    })
    print(f"wrote {len(output)} rows: {len(output)-30} train, 30 validation")


if __name__ == "__main__":
    prepare()
