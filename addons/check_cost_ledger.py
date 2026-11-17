"""Offline checks for the cost ledger and weekly budget. No network and no model call."""
from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cost_ledger as cl


def check_disabled_weekly_budget() -> None:
    policy = cl.budget_policy({"weekly_budget_usd": 25, "budget": {"enabled": False}})
    assert policy["weekly_usd"] == 25
    assert policy["enabled"] is False
    result = cl.gate(1000, 500, policy["weekly_usd"], policy["enabled"])
    assert result["ok"] is True and result["enabled"] is False
    enabled = cl.gate(24, 2, 25, True)
    assert enabled["ok"] is False and enabled["enabled"] is True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        targets_path = root / "targets.json"
        targets_path.write_text(json.dumps({"weekly_budget_usd": 25, "budget": {"enabled": False}}))
        evaluated = cl.evaluate(root, datetime.now(timezone.utc).date(), "manifest",
                                json.loads(targets_path.read_text()), 1000)
        assert evaluated["ok"] is True and evaluated["enabled"] is False
        command = [sys.executable, str(Path(cl.__file__)), "--data-root", str(root),
                   "--kind", "manifest", "--next", "1000"]
        disabled = subprocess.run(command, text=True, capture_output=True)
        assert disabled.returncode == 0, disabled.stderr
        assert "limit=disabled" in disabled.stdout and "ok=true" in disabled.stdout
        targets_path.write_text(json.dumps({"weekly_budget_usd": 25, "budget": {"enabled": True}}))
        enabled = subprocess.run(command, text=True, capture_output=True)
        assert enabled.returncode == 4, enabled.stdout + enabled.stderr
        assert "limit=$25.00" in enabled.stdout and "ok=false" in enabled.stdout


def main() -> int:
    rows = cl.load_models()
    newly_listed = {
        "deepseek/deepseek-v4.1-flash-1017",
        "moonshotai/kimi-k3",
        "moonshotai/kimi-k3.1-pro",
        "google/gemini-4-argon",
        "openai/gpt-6.1-terra",
        "openai/gpt-6-astra",
    }
    model_ids = {row["id"] for row in rows}
    assert newly_listed <= model_ids
    for row in rows:
        if row["id"] in newly_listed:
            assert row.get("usd_per_mtok_in") is None and row.get("usd_per_mtok_out") is None
    assert cl.price_tokens("openai/gpt-6.1-sol", 1_000_000, 0, rows) == 2.5
    assert cl.price_tokens("openai/gpt-6-luna", 1_000_000, 1_000_000, rows) == 0.625
    assert cl.price_tokens("anthropic/claude-opus-5.5", 1_000_000, 1_000_000, rows) == 24.0
    provider = cl.price_call("openai/gpt-6.1-sol", 100, 100, 0.01, rows)
    assert provider["cost_usd"] == 0.01 and provider["cost_source"] == "provider"
    table = cl.cost_from_usage("openai/gpt-6.1-sol", {"prompt_tokens": 2_000_000, "completion_tokens": 0}, rows)
    assert table["cost_usd"] == 5.0 and table["cost_source"] == "price_table"
    assert cl.price_tokens("missing/model", 100, 100, rows) is None

    # Model-call labels contain slashes; filenames must stay in the ledger folder,
    # while JSON retains the original label and weekly accounting sees each call once.
    label_root = Path(tempfile.mkdtemp(prefix="cost_ledger_labels_"))
    labels = ["fm/draft", "risk/aggressive r1", "NVDA", "../../outside"]
    expected_prefixes = ["fm_draft", "risk_aggressive_r1", "NVDA", "outside"]
    for index, (label, expected_prefix) in enumerate(zip(labels, expected_prefixes), start=1):
        written = cl.write_chat_cost(label_root, label=label, model="openai/gpt-6.1-sol",
                                     usage={"prompt_tokens": 0, "completion_tokens": 0,
                                            "cost": index / 10},
                                     analysis_date="2026-11-09")
        assert written is not None and written.parent == cl.costs_dir(label_root), written
        assert written.name.startswith(expected_prefix), written.name
        saved = json.loads(written.read_text(encoding="utf-8"))
        assert saved["label"] == label and saved["ticker"] == label, saved
    label_records = cl.load_cost_records(label_root)
    assert len(label_records) == len(labels), label_records
    label_week = cl.weekly_spend(label_root, datetime.now(timezone.utc).date())
    assert label_week["spent_usd"] == 1.0 and len(label_week["runs"]) == len(labels), label_week

    # Journal reconciliation is part of the budget read path. Complete zero-token
    # usage is accepted; torn and incomplete responses fail closed with line context.
    journal_root = Path(tempfile.mkdtemp(prefix="cost_ledger_journal_"))
    journal_dir = journal_root / "fund_manager" / "2026-11-17" / "fixture"
    journal_dir.mkdir(parents=True)
    journal = journal_dir / "response_attempts.jsonl"
    complete = {"event": "response", "attempt_id": "zero-usage-attempt", "label": "fm/draft",
                "model_returned": "openai/gpt-6.1-sol", "created_utc": "2026-11-17T12:00:00Z",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0}}
    missing = {"event": "response", "label": "fm/final", "model_returned": "openai/gpt-6.1-sol",
               "created_utc": "2026-11-17T12:01:00Z", "usage": {}}
    journal.write_text("{\"event\":\"response\",\n" + json.dumps(missing) + "\n"
                       + json.dumps(complete) + "\n", encoding="utf-8")
    reconciled = cl.reconcile_attempt_journals(journal_root)
    assert reconciled["response_count"] == 2
    assert len(reconciled["failures"]) == 2, reconciled
    assert all(str(journal) in failure and any(f":{line_no}:" in failure for line_no in (1, 2))
               for failure in reconciled["failures"]), reconciled["failures"]
    assert any("usage.prompt_tokens" in failure and "usage.completion_tokens" in failure
               for failure in reconciled["failures"]), reconciled["failures"]
    assert len(list(cl.costs_dir(journal_root).glob("attempt_*.json"))) == 1
    fail_closed = cl.evaluate(journal_root, datetime(2026, 11, 17).date(), "fund_manager", next_usd=0)
    assert fail_closed["ok"] is False and fail_closed["reconciliation_failures"], fail_closed

    corrupt_root = Path(tempfile.mkdtemp(prefix="cost_ledger_corrupt_attempt_"))
    corrupt_dir = corrupt_root / "fund_manager" / "2026-11-17" / "fixture"
    corrupt_dir.mkdir(parents=True)
    corrupt_journal = corrupt_dir / "response_attempts.jsonl"
    attempts = []
    for attempt_id in ("unreadable-ledger-attempt", "mismatched-ledger-attempt"):
        attempts.append({"event":"response", "attempt_id":attempt_id, "label":"fm/draft",
                         "model_returned":"openai/gpt-6.1-sol", "created_utc":"2026-11-17T13:00:00Z",
                         "usage":{"prompt_tokens":100,"completion_tokens":20,"cost":0.2}})
    corrupt_journal.write_text("".join(json.dumps(row) + "\n" for row in attempts), encoding="utf-8")
    corrupt_cost_dir = cl.costs_dir(corrupt_root)
    corrupt_cost_dir.mkdir(parents=True)
    unreadable_path = corrupt_cost_dir / ("attempt_" + hashlib.sha256(attempts[0]["attempt_id"].encode()).hexdigest() + ".json")
    unreadable_bytes = b"{truncated ledger record"
    unreadable_path.write_bytes(unreadable_bytes)
    mismatch_path = corrupt_cost_dir / ("attempt_" + hashlib.sha256(attempts[1]["attempt_id"].encode()).hexdigest() + ".json")
    mismatch_path.write_text(json.dumps({"attempt_id":"different-attempt", "cost_usd":0}), encoding="utf-8")
    corrupt_reconcile = cl.reconcile_attempt_journals(corrupt_root)
    assert len(corrupt_reconcile["failures"]) == 2, corrupt_reconcile
    assert any("unreadable" in failure for failure in corrupt_reconcile["failures"])
    assert any("does not match journaled usage" in failure for failure in corrupt_reconcile["failures"])
    assert unreadable_path.read_bytes() == unreadable_bytes, "reconciliation must not overwrite corrupt ledgers"
    corrupt_budget = cl.evaluate(corrupt_root, datetime(2026, 11, 17).date(), "fund_manager", next_usd=0)
    assert corrupt_budget["ok"] is False and corrupt_budget["reconciliation_failures"], corrupt_budget

    repriced_root = Path(tempfile.mkdtemp(prefix="cost_ledger_repriced_attempt_"))
    repriced_dir = repriced_root / "fund_manager" / "2026-11-17" / "fixture"
    repriced_dir.mkdir(parents=True)
    repriced_journal = repriced_dir / "response_attempts.jsonl"
    repriced_usage = {"prompt_tokens":1000,"completion_tokens":200}
    original_pricing = {"model":"test/model", "tokens_in":1000, "tokens_out":200,
                        "cost_usd":0.5, "cost_source":"price_table"}
    with patch.object(cl, "cost_from_usage", return_value=original_pricing):
        repriced_record = cl.write_chat_cost(repriced_root, label="fm/draft", model="test/model",
                    usage=repriced_usage, analysis_date="2026-11-17", attempt_id="repriced-attempt",
                    generated_utc="2026-11-17T14:00:00Z")
    saved_before = repriced_record.read_bytes()
    repriced_journal.write_text(json.dumps({"event":"response", "attempt_id":"repriced-attempt",
        "label":"fm/draft", "model_returned":"test/model", "analysis_date":"2026-11-17",
        "created_utc":"2026-11-17T14:00:00Z", "usage":repriced_usage}) + "\n", encoding="utf-8")
    changed_pricing = {**original_pricing, "cost_usd":9.99}
    with patch.object(cl, "cost_from_usage", return_value=changed_pricing):
        repriced = cl.reconcile_attempt_journals(repriced_root)
        spend_after_reprice = cl.weekly_spend(repriced_root, datetime(2026, 11, 17).date())
    assert not repriced["failures"] and repriced["reconciled_count"] == 1, repriced
    assert not spend_after_reprice["reconciliation_failures"], spend_after_reprice
    assert repriced_record.read_bytes() == saved_before, "replay must preserve the original estimate"
    assert json.loads(repriced_record.read_text())["cost_usd"] == 0.5

    root = Path(tempfile.mkdtemp(prefix="cost_ledger_"))
    targets = {"budget": {"weekly_usd": 25, "native_run_estimate_usd": 2, "fund_manager_estimate_usd": 1.5}}
    (root / "targets.json").write_text(json.dumps(targets), encoding="utf-8")
    folder = cl.costs_dir(root)
    folder.mkdir(parents=True)
    stamp = datetime(2026, 11, 6, 15, 0, tzinfo=timezone.utc).isoformat()
    (folder / "NVDA_2026-11-06_a.json").write_text(json.dumps({
        "ticker": "NVDA", "generated_utc": stamp, "cost_usd": 20, "cost_source": "price_table",
    }), encoding="utf-8")
    open_gate = cl.evaluate(root, datetime(2026, 11, 6).date(), "native", targets)
    assert open_gate["ok"] and open_gate["spent_usd"] == 20 and open_gate["next_usd"] == 2, open_gate
    blocked = cl.evaluate(root, datetime(2026, 11, 6).date(), "native", targets, next_usd=6)
    assert blocked["ok"] is False and blocked["projected_usd"] == 26, blocked
    (folder / "FM_2026-11-06_b.json").write_text(json.dumps({
        "label": "fm", "generated_utc": stamp, "cost_usd": 5, "cost_source": "provider",
    }), encoding="utf-8")
    full = cl.evaluate(root, datetime(2026, 11, 6).date(), "fund_manager", targets)
    assert full["ok"] is False and full["spent_usd"] == 25, full
    other_week = cl.weekly_spend(root, datetime(2026, 11, 16).date(), targets)
    assert other_week["spent_usd"] == 0, other_week
    code = cl.main(["--data-root", str(root), "--date", "2026-11-06", "--kind", "native"])
    assert code == 4
    assert cl.budget_policy({"weekly_budget_usd": 40, "budget": {"weekly_usd": 25}})["weekly_usd"] == 40
    assert cl.budget_policy({})["weekly_usd"] == 25

    # A token-only file with no USD counts at the per-run estimate in the gate.
    root2 = Path(tempfile.mkdtemp(prefix="cost_ledger_"))
    folder2 = cl.costs_dir(root2)
    folder2.mkdir(parents=True)
    (folder2 / "V_2026-11-06_a.json").write_text(json.dumps({
        "ticker": "V", "generated_utc": stamp, "cost_usd": None, "tokens_in": 900000, "tokens_out": 40000,
    }), encoding="utf-8")
    (folder2 / "X_2026-11-06_b.json").write_text(json.dumps({
        "ticker": "X", "generated_utc": stamp, "cost_usd": None,
    }), encoding="utf-8")
    week2 = cl.weekly_spend(root2, datetime(2026, 11, 6).date())
    # 0.9M in x 2.5 + 0.04M out x 10 at the fallback model
    assert week2["spent_usd"] == 2.65 and week2["estimated_runs"] == 1 and week2["unpriced_runs"] == 1, week2
    unp = cl.evaluate(root2, datetime(2026, 11, 6).date(), "native", {"weekly_budget_usd": 6})
    assert unp["projected_usd"] == 6.65 and unp["unpriced_estimate_usd"] == 2 and unp["ok"] is False, unp
    # run_native style record: summed call costs
    rec = cl.price_call("openai/gpt-6-luna", 400_000, 20_000, None, rows)
    assert rec["cost_source"] == "price_table" and rec["cost_usd"] == 0.06, rec
    fb = cl.price_call(None, 1_000_000, 0, None, rows, fallback_model="openai/gpt-6.1-sol")
    assert fb["cost_source"] == "price_table_fallback" and fb["cost_usd"] == 2.5, fb
    assert cl.price_call(None, 0, 0, None, rows, fallback_model="openai/gpt-6.1-sol")["cost_usd"] is None
    print("check_cost_ledger: ok")
    print(f"  open projected {open_gate['projected_usd']} blocked projected {blocked['projected_usd']}")
    return 0


if __name__ == "__main__":
    check_disabled_weekly_budget()
    sys.exit(main())
