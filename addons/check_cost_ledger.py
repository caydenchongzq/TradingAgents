"""Offline checks for the cost ledger and weekly budget. No network and no model call."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

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
