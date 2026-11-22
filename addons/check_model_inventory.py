#!/usr/bin/env python3
"""Offline checks for model metadata normalization and atomic cache writes."""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_inventory as inventory


def main() -> int:
    payload = {"data": [{
        "id": "openai/gpt-6.2-sol",
        "name": "GPT-6.2 Sol",
        "description": "Example model",
        "context_length": 272000,
        "top_provider": {"max_completion_tokens": 12000},
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "supported_parameters": ["tools", "json_schema"],
        "pricing": {"prompt": "0.000002", "completion": "0.00001", "image": "0.003",
                    "overrides": [{"min_prompt_tokens": 272000, "prompt": "0.000004",
                                   "completion": "0.000015"}]},
    }]}
    snapshot = inventory.normalize_openrouter(payload, fetched_at="2026-11-21T12:00:00Z")
    row = snapshot["models"][0]
    assert row["id"] == "openai/gpt-6.2-sol"
    assert row["usd_per_mtok_in"] == 2 and row["usd_per_mtok_out"] == 10
    assert row["context_length"] == 272000 and row["max_output_tokens"] == 12000
    assert row["input_modalities"] == ["text"] and row["output_modalities"] == ["text"]
    assert row["pricing"]["image"] == "0.003"
    assert inventory.compatible_seats(row)
    assert "market" in inventory.compatible_seats(row)
    assert "fm_manager" in inventory.compatible_seats(row)
    assert inventory.effective_rates([row["id"]], snapshot) == [{
        "id": row["id"], "available": True, "usd_per_mtok_in": 2.0,
        "usd_per_mtok_out": 10.0,
        "price_tiers": [{"min_prompt_tokens": 272000,
                          "usd_per_mtok_in": 4.0, "usd_per_mtok_out": 15.0}],
        "input_modalities": ["text"], "output_modalities": ["text"],
        "supported_parameters": ["json_schema", "tools"],
        "has_conditional_pricing": False,
        "pricing": {"prompt": 2, "completion": 10},
    }]
    tiered = inventory.normalize_openrouter({"data": [{
        "id": "openai/gpt-6.2-astra", "name": "GPT-6.2 Astra",
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "supported_parameters": ["tools", "structured_outputs"],
        "pricing": {"prompt": "0.000008", "completion": "0.00004",
                    "overrides": [{"min_prompt_tokens": 272000, "prompt": "0.000016",
                                   "completion": "0.00006", "input_cache_read": "0.000001"}],
                    "input_cache_read": "0.000001"},
    }]})["models"][0]
    assert tiered["price_tiers"] == [{"min_prompt_tokens": 272000,
                                      "usd_per_mtok_in": 16,
                                      "usd_per_mtok_out": 60}]
    assert tiered["pricing"]["input_cache_read"] == "0.000001"
    assert inventory.token_rates(tiered, 272000) == (16.0, 60.0)
    scheduled = inventory.normalize_openrouter({"data": [{
        "id": "minimax/scheduled", "architecture": {
            "input_modalities": ["text"], "output_modalities": ["text"]},
        "supported_parameters": ["tools"],
        "pricing": {"prompt": "0.00001", "completion": "0.00002", "overrides": [
            {"utc_start": "00:00", "utc_end": "12:00", "prompt": "0.00004",
             "completion": "0.00008"},
            {"utc_days": ["mon", "tue"], "min_prompt_tokens": 100000,
             "prompt": "0.00003", "completion": "0.00006"},
        ]},
    }]})["models"][0]
    assert scheduled["price_tiers"] == []
    assert scheduled["has_conditional_pricing"] is True
    assert inventory.token_rates(scheduled, 200000) == (10.0, 20.0)
    assert inventory.effective_rates(["minimax/scheduled"], {"models": [scheduled]})[0][
        "has_conditional_pricing"] is True
    try:
        inventory.normalize_openrouter({"data": [{"id": "provider/bad-threshold", "pricing": {
            "prompt": "0.00001", "completion": "0.00002",
            "overrides": [{"utc_days": ["mon"], "min_prompt_tokens": "bad",
                           "prompt": "0.00003", "completion": "0.00004"}]}}]})
    except inventory.InventoryError:
        pass
    else:
        raise AssertionError("malformed threshold on a scheduled override was accepted")
    try:
        inventory.normalize_openrouter({"data": [{"id": "provider/bad-pricing-type", "pricing": []}]})
    except inventory.InventoryError:
        pass
    else:
        raise AssertionError("non-object pricing was accepted")
    try:
        inventory.normalize_openrouter({"data": [{"id": "provider/nested-nan",
                                                   "pricing": {"prompt": "0.00001",
                                                               "completion": "0.00002"},
                                                   "provider_metadata": {"unexpected": float("nan")}}]})
    except inventory.InventoryError:
        pass
    else:
        raise AssertionError("nested NaN provider metadata was accepted")
    router = inventory.normalize_openrouter({"data": [{
        "id": "openrouter/auto", "name": "OpenRouter Auto",
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "supported_parameters": ["tools"],
        "pricing": {"prompt": "-1", "completion": "-1"},
    }]})["models"][0]
    assert router["usd_per_mtok_in"] is None and router["usd_per_mtok_out"] is None
    assert router["pricing"]["prompt"] is None and router["pricing"]["completion"] is None
    assert inventory.effective_rates(["openrouter/auto"], {"models": [router]})[-1]["pricing"] == {
        "prompt": None, "completion": None,
    }
    assert inventory.normalize_openrouter({"data": [{
        "id": "meta/llama-3.1-8b-instruct:free", "name": "Free model",
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "supported_parameters": ["tools"],
        "pricing": {"prompt": "0", "completion": "0"},
    }]})["models"][0]["id"] == "meta/llama-3.1-8b-instruct:free"
    for bad_rate in ("1e308", "-2", "malformed", True):
        try:
            inventory.normalize_openrouter({"data": [{"id": "publisher/invalid-rate", "pricing": {
                "prompt": bad_rate, "completion": "0"}}]})
        except inventory.InventoryError:
            pass
        else:
            raise AssertionError(f"invalid rate {bad_rate!r} was accepted")
    for bad_tier_rate in ("malformed", True):
        try:
            inventory.normalize_openrouter({"data": [{"id": "publisher/invalid-tier-rate", "pricing": {
                "prompt": "0.000001", "completion": "0.000002",
                "overrides": [{"min_prompt_tokens": 1000,
                               "prompt": bad_tier_rate, "completion": "0.000003"}]}}]})
        except inventory.InventoryError:
            pass
        else:
            raise AssertionError(f"invalid tier rate {bad_tier_rate!r} was accepted")

    assert inventory.token_rates({"usd_per_mtok_in": 2, "usd_per_mtok_out": 10,
                                  "price_tiers": [{"min_prompt_tokens": 272000,
                                                   "usd_per_mtok_in": 4,
                                                   "usd_per_mtok_out": 15}]}, 271999) == (2, 10)
    assert inventory.token_rates({"usd_per_mtok_in": 2, "usd_per_mtok_out": 10,
                                  "price_tiers": [{"min_prompt_tokens": 272000,
                                                   "usd_per_mtok_in": 4,
                                                   "usd_per_mtok_out": 15}]}, 272000) == (4.0, 15.0)
    fixed_now = datetime(2026, 11, 21, 12, tzinfo=timezone.utc)
    assert not inventory.is_stale("2026-11-20T12:00:00Z", now=fixed_now)
    assert inventory.is_stale("2026-11-20T11:59:59Z", now=fixed_now)
    assert inventory.is_stale("2026-11-21T13:00:00Z", now=fixed_now)
    assert inventory.is_stale("not-a-time", now=fixed_now)
    assert inventory.is_stale("2020-01-01T00:00:00Z", now=fixed_now)
    assert inventory.compatible_seats({"id": "publisher/image-model", "input_modalities": ["image"],
                                       "output_modalities": ["text"], "supported_parameters": ["tools"]}) == []
    assert inventory.compatible_seats({"id": "publisher/text-image-out", "input_modalities": ["text"],
                                       "output_modalities": ["image"], "supported_parameters": ["tools"]}) == []
    unknown = inventory.normalize_row({"id": "publisher/unknown", "name": "Unknown"})
    assert unknown["compatibility_unknown"] and inventory.compatible_seats(unknown) == []

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "catalog.json"
        path.write_text(json.dumps({"models": [{"id": "legacy/model", "label": "Legacy",
                                                  "usd_per_mtok_in": 1, "usd_per_mtok_out": 2}]}))
        legacy = inventory.load_snapshot(path)
        assert legacy["stale"] and legacy["models"][0]["id"] == "legacy/model"
        inventory.write_snapshot(path, snapshot)
        cached = inventory.load_snapshot(path)
        assert cached["fetched_at"] == "2026-11-21T12:00:00Z"
        retired = next(row for row in cached["models"] if row["id"] == "legacy/model")
        assert retired["available"] is False and retired["usd_per_mtok_in"] == 1
        previous_tier = next(row for row in cached["models"] if row["id"] == "openai/gpt-6.2-sol")
        assert previous_tier["price_tiers"]
        fresh_rate = inventory.normalize_openrouter({"data": [{"id": "openai/gpt-6.2-sol",
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_parameters": ["tools"],
            "pricing": {"prompt": "0.000002", "completion": "1e-5"}}]})
        inventory.write_snapshot(path, fresh_rate)
        refreshed = inventory.load_snapshot(path)
        replaced = next(row for row in refreshed["models"] if row["id"] == "openai/gpt-6.2-sol")
        assert replaced["price_tiers"] == []
        equivalent = inventory.normalize_openrouter({"data": [{"id": "openai/gpt-6.2-sol",
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_parameters": ["tools"],
            "pricing": {"prompt": "0.0000020", "completion": "0.0000100"}}]})
        assert inventory.effective_rates(["openai/gpt-6.2-sol"], refreshed) == inventory.effective_rates(
            ["openai/gpt-6.2-sol"], equivalent)
        back = inventory.normalize_openrouter({"data": [{"id": "legacy/model", "name": "Legacy v2",
            "usd_per_mtok_in": 2, "usd_per_mtok_out": 3, "input_modalities": ["text"],
            "supported_parameters": ["tools"]}]})
        inventory.write_snapshot(path, back)
        returned = next(row for row in inventory.load_snapshot(path)["models"] if row["id"] == "legacy/model")
        assert returned["available"] is True and returned["availability_status"] == "available"
        before = path.read_bytes()
        try:
            inventory.write_snapshot(path, {"models": []})
        except inventory.InventoryError:
            pass
        else:
            raise AssertionError("empty update unexpectedly replaced the catalog")
        assert path.read_bytes() == before
    try:
        inventory.normalize_openrouter({"data": []})
    except inventory.InventoryError:
        pass
    else:
        raise AssertionError("empty provider response was accepted")
    print("model inventory checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
