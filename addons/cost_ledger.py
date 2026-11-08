#!/usr/bin/env python3
"""OpenRouter spend for add-on runs. No model call and no API key.

Provider ``usage.cost`` wins when a response carries it. Otherwise tokens are
priced from addons/openrouter_models.json. Per-run JSON files live in
``<data-root>/fund_manager/run_costs``. The weekly gate reads those files.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

DEFAULT_WEEKLY_USD = 25.0
DEFAULT_NATIVE_ESTIMATE = 2.0
DEFAULT_FUND_ESTIMATE = 1.5
DEFAULT_FALLBACK_MODEL = "openai/gpt-6.1-sol"


def models_path() -> Path:
    return Path(__file__).resolve().parent / "openrouter_models.json"


def load_models(path: Path | None = None) -> list[dict]:
    file = path or models_path()
    if not file.is_file():
        return []
    data = json.loads(file.read_text(encoding="utf-8"))
    rows = data.get("models") if isinstance(data, dict) else data
    return [row for row in rows or [] if isinstance(row, dict)]


def match_model(model_id: str | None, rows: list[dict] | None = None) -> dict | None:
    model_id = str(model_id or "").strip()
    if not model_id:
        return None
    rows = rows if rows is not None else load_models()
    exact = [row for row in rows if row.get("id") == model_id]
    if exact:
        return exact[0]
    prefix = [row for row in rows if model_id.startswith(str(row.get("id") or ""))]
    if prefix:
        return max(prefix, key=lambda row: len(str(row.get("id") or "")))
    return None


def price_tokens(model_id: str | None, tokens_in, tokens_out, rows: list[dict] | None = None) -> float | None:
    row = match_model(model_id, rows)
    if row is None:
        return None
    pin = row.get("usd_per_mtok_in")
    pout = row.get("usd_per_mtok_out")
    if not isinstance(pin, (int, float)) or not isinstance(pout, (int, float)):
        return None
    inn = float(tokens_in or 0)
    out = float(tokens_out or 0)
    return round(inn / 1_000_000 * float(pin) + out / 1_000_000 * float(pout), 6)


def price_call(model_id: str | None, tokens_in, tokens_out, provider_cost=None, rows: list[dict] | None = None,
               fallback_model: str | None = None) -> dict:
    """One completion. A provider cost is used as-is. Otherwise the price table."""
    rec = {
        "model": model_id,
        "tokens_in": int(tokens_in or 0),
        "tokens_out": int(tokens_out or 0),
        "cost_usd": None,
        "cost_source": "unpriced",
    }
    if provider_cost is not None:
        try:
            rec["cost_usd"] = round(float(provider_cost), 6)
            rec["cost_source"] = "provider"
            return rec
        except (TypeError, ValueError):
            pass
    priced = price_tokens(model_id, tokens_in, tokens_out, rows)
    if priced is not None:
        rec["cost_usd"] = priced
        rec["cost_source"] = "price_table"
    elif fallback_model and (rec["tokens_in"] or rec["tokens_out"]):
        priced = price_tokens(fallback_model, tokens_in, tokens_out, rows)
        if priced is not None:
            rec["cost_usd"] = priced
            rec["cost_source"] = "price_table_fallback"
            rec["priced_as"] = fallback_model
    return rec


def cost_from_usage(model_id: str | None, usage: dict | None, rows: list[dict] | None = None) -> dict:
    usage = usage or {}
    tokens_in = usage.get("prompt_tokens", usage.get("input_tokens"))
    tokens_out = usage.get("completion_tokens", usage.get("output_tokens"))
    return price_call(model_id, tokens_in, tokens_out, usage.get("cost"), rows)


def costs_dir(data_root: Path | str) -> Path:
    return Path(data_root) / "fund_manager" / "run_costs"


def write_run_cost(data_root: Path | str, record: dict) -> Path:
    folder = costs_dir(data_root)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%H%M%S_%f")
    ticker = str(record.get("ticker") or record.get("label") or "run")
    day = str(record.get("analysis_date") or "nodate").replace(":", "-")
    path = folder / f"{ticker}_{day}_{stamp}.json"
    body = dict(record)
    body.setdefault("generated_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def write_chat_cost(data_root: Path | str, *, label: str, model: str, usage: dict | None,
                    analysis_date: str | None = None) -> Path | None:
    priced = cost_from_usage(model, usage)
    if priced["cost_usd"] is None and not priced["tokens_in"] and not priced["tokens_out"]:
        return None
    return write_run_cost(data_root, {
        "label": label,
        "ticker": label,
        "analysis_date": analysis_date,
        "model": model,
        "completions": 1,
        "tokens_in": priced["tokens_in"],
        "tokens_out": priced["tokens_out"],
        "cost_usd": priced["cost_usd"],
        "cost_source": priced["cost_source"],
        "calls": [priced],
    })


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def week_bounds(day: date) -> tuple[datetime, datetime]:
    """Monday 00:00 UTC through the next Monday, exclusive."""
    start_day = day - timedelta(days=day.weekday())
    start = datetime(start_day.year, start_day.month, start_day.day, tzinfo=timezone.utc)
    return start, start + timedelta(days=7)


def _record_time(record: dict, path: Path) -> datetime | None:
    parsed = _parse_utc(record.get("generated_utc"))
    if parsed:
        return parsed
    parsed = _parse_utc(record.get("analysis_date"))
    if parsed:
        return parsed
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)


def load_cost_records(data_root: Path | str) -> list[dict]:
    folder = costs_dir(data_root)
    if not folder.is_dir():
        return []
    rows = []
    for path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            data["_path"] = str(path)
            rows.append(data)
    return rows


def weekly_spend(data_root: Path | str, day: date | None = None, targets: dict | None = None) -> dict:
    day = day or datetime.now(timezone.utc).date()
    start, end = week_bounds(day)
    budget = budget_policy(targets)
    spent = 0.0
    runs = []
    unpriced = 0
    estimated = 0
    for record in load_cost_records(data_root):
        when = _record_time(record, Path(record.get("_path") or "."))
        if when is None or not (start <= when < end):
            continue
        cost = record.get("cost_usd")
        source = record.get("cost_source")
        if not isinstance(cost, (int, float)) and (record.get("tokens_in") or record.get("tokens_out")):
            # Token-only file (written before USD capture). Price it at the run's model, or at
            # the fallback model, which is the dearest TradingAgents default, so it is an upper bound.
            est = price_tokens(record.get("model") or budget["fallback_model"],
                               record.get("tokens_in"), record.get("tokens_out"))
            if est is not None:
                cost, source = est, "price_table_fallback"
                estimated += 1
        if isinstance(cost, (int, float)):
            spent += float(cost)
        else:
            unpriced += 1
        runs.append({
            "label": record.get("label") or record.get("ticker"),
            "analysis_date": record.get("analysis_date"),
            "generated_utc": record.get("generated_utc"),
            "cost_usd": cost,
            "cost_source": source,
            "tokens_in": record.get("tokens_in"),
            "tokens_out": record.get("tokens_out"),
        })
    return {
        "week_start": start.date().isoformat(),
        "week_end": (end - timedelta(days=1)).date().isoformat(),
        "spent_usd": round(spent, 6),
        "unpriced_runs": unpriced,
        "estimated_runs": estimated,
        "budget_usd": budget["weekly_usd"],
        "runs": runs,
    }


def budget_policy(targets: dict | None) -> dict:
    raw = (targets or {}).get("budget") or {}
    # Top-level weekly_budget_usd wins, then budget.weekly_usd, then 25.
    weekly = (targets or {}).get("weekly_budget_usd", raw.get("weekly_usd", DEFAULT_WEEKLY_USD))
    native = raw.get("native_run_estimate_usd", DEFAULT_NATIVE_ESTIMATE)
    fund = raw.get("fund_manager_estimate_usd", DEFAULT_FUND_ESTIMATE)
    return {
        "fallback_model": str(raw.get("fallback_model") or DEFAULT_FALLBACK_MODEL),
        "weekly_usd": float(weekly),
        "native_run_estimate_usd": float(native),
        "fund_manager_estimate_usd": float(fund),
    }


def gate(spent_usd: float, next_usd: float, weekly_usd: float) -> dict:
    """Hard stop before a run that would take the week over the budget."""
    projected = float(spent_usd) + float(next_usd)
    ok = projected <= float(weekly_usd) + 1e-9
    return {
        "ok": ok,
        "spent_usd": round(float(spent_usd), 6),
        "next_usd": round(float(next_usd), 6),
        "projected_usd": round(projected, 6),
        "budget_usd": float(weekly_usd),
    }


def evaluate(data_root: Path | str, day: date, kind: str, targets: dict | None = None, next_usd: float | None = None) -> dict:
    policy = budget_policy(targets)
    spend = weekly_spend(data_root, day, targets)
    if next_usd is None:
        if kind == "native":
            next_usd = policy["native_run_estimate_usd"]
        elif kind == "fund_manager":
            next_usd = policy["fund_manager_estimate_usd"]
        else:
            next_usd = 0.0
    # A run with neither USD nor tokens counts at the per-run estimate, so a gap in
    # pricing cannot slip a week past the budget.
    unpriced_usd = spend["unpriced_runs"] * policy["native_run_estimate_usd"]
    result = gate(spend["spent_usd"] + unpriced_usd, next_usd, policy["weekly_usd"])
    result.update({"week_start": spend["week_start"], "week_end": spend["week_end"], "kind": kind,
                   "priced_usd": spend["spent_usd"], "unpriced_runs": spend["unpriced_runs"],
                   "unpriced_estimate_usd": round(unpriced_usd, 6)})
    return result


def _load_targets(data_root: Path) -> dict:
    path = data_root / "targets.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check the weekly OpenRouter budget. No model call.")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--kind", choices=["native", "fund_manager", "check"], default="check")
    ap.add_argument("--next", type=float, default=None)
    args = ap.parse_args(argv)
    root = Path(args.data_root)
    day = datetime.strptime(args.date, "%Y-%m-%d").date()
    result = evaluate(root, day, args.kind, _load_targets(root), args.next)
    print(
        f"BUDGET spent={result['spent_usd']} limit={result['budget_usd']} "
        f"next={result['next_usd']} projected={result['projected_usd']} "
        f"unpriced_runs={result['unpriced_runs']} ok={str(result['ok']).lower()}"
    )
    return 0 if result["ok"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
