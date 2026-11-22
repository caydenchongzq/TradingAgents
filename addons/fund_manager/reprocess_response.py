#!/usr/bin/env python3
"""Re-parse one saved fund-manager response without contacting a model provider.

Writes a separate recovery JSON beside the source report and never changes source
artifacts, run metadata, latest pointers, or cost records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fund_manager as fm

SCHEMAS = {
    "fund_manager_draft": fm.LEGACY_DRAFT_SCHEMA,
    "fund_manager_final": fm.LEGACY_FINAL_SCHEMA,
    "fund_manager_draft_v2": fm.DRAFT_SCHEMA,
    "fund_manager_final_v2": fm.FINAL_SCHEMA,
    "aggressive_risk_view": fm.ANALYST_SCHEMA,
    "conservative_risk_view": fm.ANALYST_SCHEMA,
    "neutral_risk_view": fm.ANALYST_SCHEMA,
}


def _responses(source: dict, journal: Path) -> list[dict]:
    responses = []
    if journal.is_file():
        records = {}
        statuses = {}
        for line in journal.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (row.get("label"), row.get("attempt"))
            if row.get("event") == "response":
                records[key] = row
            elif row.get("event") == "parse_result":
                statuses[key] = row
        for key, row in records.items():
            responses.append({**row, "parse_result": statuses.get(key)})
    if responses:
        return responses
    # Legacy reports only saved free-text fallbacks. Keep those inspectable even
    # when their prose cannot be converted to the strict decision schema.
    for index, content in enumerate(source.get("raw_text_fallbacks") or []):
        responses.append({"event": "legacy_text", "attempt": index + 1,
                          "label": "fm/draft", "schema_name": "fund_manager_draft",
                          "mode": "text", "content": content})
    return responses


def _reconcile_cost(report_path: Path, report: dict, attempt: dict) -> str:
    usage = attempt.get("usage")
    model = attempt.get("model_returned") or attempt.get("model_requested")
    attempt_id = attempt.get("attempt_id")
    if not usage or not model or not attempt_id:
        return "unavailable"
    try:
        # Report path is <data>/fund_manager/<date>/<run>/portfolio_report.json.
        data_root = report_path.parents[3]
        addons_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(addons_root))
        import cost_ledger
        written = cost_ledger.write_chat_cost(data_root, label=attempt.get("label", "fund_manager"),
                                              model=model, usage=usage,
                                              analysis_date=report.get("date") or attempt.get("analysis_date"),
                                              attempt_id=attempt_id, generated_utc=attempt.get("created_utc"))
        return "reconciled" if written else "no_billable_usage"
    except Exception as exc:
        return f"failed: {str(exc)[:300]}"


def reprocess(report_path: Path, index: int) -> Path:
    report_path = report_path.resolve()
    original = report_path.read_bytes()
    journal_only = report_path.name == "response_attempts.jsonl"
    if journal_only:
        report = {}
        journal = report_path
        journal_bytes = original
    else:
        report = json.loads(original)
        journal = report_path.parent / "response_attempts.jsonl"
        journal_bytes = journal.read_bytes() if journal.is_file() else b""
    attempts = _responses(report, journal)
    if index < 1 or index > len(attempts):
        raise ValueError(f"attempt must be between 1 and {len(attempts)}")
    attempt = attempts[index - 1]
    schema_name = attempt.get("schema_name")
    schema = SCHEMAS.get(schema_name)
    result = {"source_report": None if journal_only else report_path.name,
              "source_artifact": report_path.name,
              "source_artifact_type": "response_journal" if journal_only else "report",
              "source_sha256": hashlib.sha256(original).hexdigest(),
              "response_journal": journal.name if journal_bytes else None,
              "response_journal_sha256": hashlib.sha256(journal_bytes).hexdigest() if journal_bytes else None,
              "selected_response_sha256": hashlib.sha256(str(attempt.get("content", "")).encode("utf-8")).hexdigest(),
              "attempt_index": index,
              "attempt_label": attempt.get("label"),
              "attempt_mode": attempt.get("mode"),
              "attempt_response_id": attempt.get("response_id"),
              "attempt_created_utc": attempt.get("created_utc"),
              "parser": "fund_manager.parse_structured_response",
              "schema_name": schema_name,
              "reprocessed_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
              "status": "unavailable", "content": attempt.get("content", ""),
              "cost_reconciliation": _reconcile_cost(report_path, report, attempt)}
    if not schema:
        result["reason"] = "No registered schema for this saved response."
    else:
        try:
            parsed = fm.parse_structured_response(result["content"], schema)
            result["status"] = "schema_valid"
            result["parsed_candidate"] = parsed
            if schema_name in ("fund_manager_draft", "fund_manager_final",
                               "fund_manager_draft_v2", "fund_manager_final_v2"):
                if journal_only or not isinstance(report.get("stats"), dict):
                    result["trade_validation_status"] = "not_run_missing_report_stats"
                else:
                    try:
                        result["trade_validation"] = fm.validate_trades(parsed, report["stats"])
                        result["status"] = "trade_validated_candidate"
                    except Exception as exc:
                        result["trade_validation_error"] = str(exc)[:1000]
        except Exception as exc:
            result["status"] = "unavailable"
            result["reason"] = str(exc)[:1000]
    digest = hashlib.sha256((str(index) + result["source_sha256"] + result["reprocessed_utc"]).encode()).hexdigest()[:12]
    destination = report_path.parent / f"recovery_{digest}.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, default=str)
        stream.write("\n")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path,
                        help="Saved portfolio_report.json, portfolio_snapshot.json, or response_attempts.jsonl")
    parser.add_argument("--attempt", type=int, default=1, help="1-based response attempt index")
    args = parser.parse_args()
    try:
        output = reprocess(args.source, args.attempt)
    except Exception as exc:
        parser.error(str(exc))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
