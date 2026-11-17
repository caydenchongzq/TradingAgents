#!/usr/bin/env python3
"""OpenRouter spend for add-on runs. No model call and no API key.

Provider ``usage.cost`` wins when a response carries it. Otherwise tokens are
priced from addons/openrouter_models.json. Per-run JSON files live in
``<data-root>/fund_manager/run_costs``. The weekly gate reads those files.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import uuid
import json
import math
import re
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


def _filename_component(value: object) -> str:
    """Return a single safe filename component while keeping ordinary tickers intact."""
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "run")).strip("._-")
    return component or "run"


def _validate_existing_attempt_record(path: Path, expected: dict) -> Path:
    """Check immutable response identity without repricing a durable estimate."""
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"existing attempt ledger record is unreadable: {path}") from exc
    immutable = ("attempt_id", "label", "ticker", "analysis_date", "model", "completions",
                 "tokens_in", "tokens_out", "generated_utc")
    if not isinstance(existing, dict) or any(existing.get(key) != expected.get(key) for key in immutable):
        raise ValueError(f"existing attempt ledger record does not match journaled usage: {path}")
    # New records preserve the full provider usage object. Older attempt records
    # predate this field; their call identity still has to match below.
    if "raw_usage" in existing and existing["raw_usage"] != expected.get("raw_usage"):
        raise ValueError(f"existing attempt ledger raw usage does not match journal: {path}")
    calls = existing.get("calls")
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        raise ValueError(f"existing attempt ledger call details are invalid: {path}")
    call = calls[0]
    for key in ("model", "tokens_in", "tokens_out"):
        if call.get(key) != expected.get(key):
            raise ValueError(f"existing attempt ledger {key} does not match journal: {path}")
    if existing.get("cost_source") not in {"provider", "price_table", "price_table_fallback", "unpriced"}:
        raise ValueError(f"existing attempt ledger cost source is invalid: {path}")
    cost = existing.get("cost_usd")
    if (cost is not None and (isinstance(cost, bool) or not isinstance(cost, (int, float))
                              or (isinstance(cost, float) and not math.isfinite(cost)) or cost < 0)):
        raise ValueError(f"existing attempt ledger cost is invalid: {path}")
    if call.get("cost_usd") != cost or call.get("cost_source") != existing.get("cost_source"):
        raise ValueError(f"existing attempt ledger cost details are inconsistent: {path}")
    # A provider supplied price is immutable. Estimated prices are intentionally
    # retained as first recorded because the price table can change over time.
    raw_usage = expected.get("raw_usage") or {}
    provider_cost = raw_usage.get("cost") if isinstance(raw_usage, dict) else None
    if (isinstance(provider_cost, (int, float)) and not isinstance(provider_cost, bool)
            and (existing.get("cost_source") != "provider" or cost != round(float(provider_cost), 6))):
        raise ValueError(f"existing attempt ledger provider cost does not match journal: {path}")
    return path


def write_run_cost(data_root: Path | str, record: dict) -> Path:
    folder = costs_dir(data_root)
    folder.mkdir(parents=True, exist_ok=True)
    body = dict(record)
    body.setdefault("generated_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    attempt_id = body.get("attempt_id")
    if attempt_id:
        digest = hashlib.sha256(str(attempt_id).encode("utf-8")).hexdigest()
        path = folder / f"attempt_{digest}.json"
        if path.exists():
            return _validate_existing_attempt_record(path, body)
        temp = folder / f".{digest}.{uuid.uuid4().hex}.tmp"
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(body, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temp, path)
            except FileExistsError:
                # A concurrent writer may have completed first. Verify the file
                # before treating this replay as safely accounted.
                return _validate_existing_attempt_record(path, body)
            return path
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
    stamp = datetime.now(timezone.utc).strftime("%H%M%S_%f")
    ticker = _filename_component(body.get("ticker") or body.get("label") or "run")
    day = _filename_component(str(body.get("analysis_date") or "nodate").replace(":", "-"))
    path = folder / f"{ticker}_{day}_{stamp}.json"
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def write_chat_cost(data_root: Path | str, *, label: str, model: str, usage: dict | None,
                    analysis_date: str | None = None, attempt_id: str | None = None,
                    generated_utc: str | None = None) -> Path | None:
    priced = cost_from_usage(model, usage)
    if priced["cost_usd"] is None and not priced["tokens_in"] and not priced["tokens_out"]:
        return None
    record = {
        "label": label,
        "ticker": label,
        "analysis_date": analysis_date,
        "attempt_id": attempt_id,
        "model": model,
        "completions": 1,
        "tokens_in": priced["tokens_in"],
        "tokens_out": priced["tokens_out"],
        "raw_usage": usage if isinstance(usage, dict) else {},
        "cost_usd": priced["cost_usd"],
        "cost_source": priced["cost_source"],
        "calls": [priced],
    }
    if generated_utc is not None:
        record["generated_utc"] = generated_utc
    return write_run_cost(data_root, record)


def reconcile_attempt_journals(data_root: Path | str) -> dict:
    """Idempotently account journaled model completions before budget gates.

    Completion timestamps are carried into cost records so a delayed recovery
    remains charged to the week in which the provider returned the response.
    Malformed or incomplete records fail closed with their journal and line.
    """
    root = Path(data_root)
    journals = sorted((root / "fund_manager").glob("**/response_attempts.jsonl"))
    response_count = reconciled = 0
    failures = []
    for journal in journals:
        try:
            lines = journal.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            failures.append(f"{journal}: {exc}")
            continue
        for number, line in enumerate(lines, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                failures.append(f"{journal}:{number}: malformed JSON journal line")
                continue
            if not isinstance(row, dict):
                failures.append(f"{journal}:{number}: journal record must be an object")
                continue
            if row.get("event") != "response":
                continue
            response_count += 1
            attempt_id = row.get("attempt_id")
            model = row.get("model_returned")
            label = row.get("label")
            usage = row.get("usage")
            created_utc = row.get("created_utc")
            problems = []
            if not isinstance(attempt_id, str) or not attempt_id.strip():
                problems.append("missing attempt_id")
            if not isinstance(model, str) or not model.strip():
                problems.append("missing model_returned")
            if not isinstance(label, str) or not label.strip():
                problems.append("missing label")
            if not isinstance(usage, dict):
                problems.append("missing usage object")
            else:
                for field in ("prompt_tokens", "completion_tokens"):
                    value = usage.get(field)
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or (isinstance(value, float) and not math.isfinite(value)) or value < 0):
                        problems.append(f"missing or invalid usage.{field}")
                if "cost" in usage:
                    cost = usage["cost"]
                    if cost is not None and (isinstance(cost, bool) or not isinstance(cost, (int, float))
                                             or (isinstance(cost, float) and not math.isfinite(cost)) or cost < 0):
                        problems.append("invalid usage.cost")
            if not isinstance(created_utc, str) or _parse_utc(created_utc) is None:
                problems.append("missing or invalid created_utc")
            if problems:
                failures.append(f"{journal}:{number}: incomplete response record ({'; '.join(problems)})")
                continue
            try:
                written = write_chat_cost(root, label=label,
                                          model=model, usage=usage,
                                          analysis_date=row.get("analysis_date"),
                                          attempt_id=attempt_id,
                                          generated_utc=created_utc)
                if written is not None:
                    reconciled += 1
            except Exception as exc:  # caller must stop before spending more
                failures.append(f"{journal}:{number}: {exc}")
    return {"journal_count": len(journals), "response_count": response_count,
            "reconciled_count": reconciled, "failures": failures}


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
    reconciliation = reconcile_attempt_journals(data_root)
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
        "budget_enabled": budget["enabled"],
        "runs": runs,
        "reconciliation_failures": reconciliation["failures"],
    }


def budget_policy(targets: dict | None) -> dict:
    raw = (targets or {}).get("budget") or {}
    # Top-level weekly_budget_usd wins, then budget.weekly_usd, then 25.
    # budget.enabled=false suspends the weekly gate without deleting the saved
    # allowance; usage and estimates continue to be recorded.
    weekly = (targets or {}).get("weekly_budget_usd", raw.get("weekly_usd", DEFAULT_WEEKLY_USD))
    native = raw.get("native_run_estimate_usd", DEFAULT_NATIVE_ESTIMATE)
    fund = raw.get("fund_manager_estimate_usd", DEFAULT_FUND_ESTIMATE)
    return {
        "fallback_model": str(raw.get("fallback_model") or DEFAULT_FALLBACK_MODEL),
        "weekly_usd": float(weekly),
        "enabled": raw.get("enabled", True) is not False,
        "native_run_estimate_usd": float(native),
        "fund_manager_estimate_usd": float(fund),
    }


def gate(spent_usd: float, next_usd: float, weekly_usd: float, enabled: bool = True) -> dict:
    """Hard stop before a run that would take the week over the budget."""
    projected = float(spent_usd) + float(next_usd)
    ok = not enabled or projected <= float(weekly_usd) + 1e-9
    return {
        "ok": ok,
        "enabled": enabled,
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
    result = gate(spend["spent_usd"] + unpriced_usd, next_usd, policy["weekly_usd"], policy["enabled"])
    result.update({"week_start": spend["week_start"], "week_end": spend["week_end"], "kind": kind,
                   "priced_usd": spend["spent_usd"], "unpriced_runs": spend["unpriced_runs"],
                   "unpriced_estimate_usd": round(unpriced_usd, 6),
                   "reconciliation_failures": spend["reconciliation_failures"]})
    if spend["reconciliation_failures"]:
        result["ok"] = False
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
    ap.add_argument("--kind", choices=["native", "fund_manager", "manifest", "check"], default="check")
    ap.add_argument("--next", type=float, default=None)
    args = ap.parse_args(argv)
    root = Path(args.data_root)
    day = datetime.strptime(args.date, "%Y-%m-%d").date()
    result = evaluate(root, day, args.kind, _load_targets(root), args.next)
    limit = f"${result['budget_usd']:.2f}" if result.get("enabled", True) else "disabled"
    print(
        f"BUDGET spent={result['spent_usd']} limit={limit} "
        f"next={result['next_usd']} projected={result['projected_usd']} "
        f"unpriced_runs={result['unpriced_runs']} ok={str(result['ok']).lower()}"
    )
    return 0 if result["ok"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
