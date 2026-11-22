"""Trustworthy cross-review context for advisory fund-manager decisions.

Only completed portfolio reports are eligible. This module never infers fills
from a later holdings snapshot and never treats a hypothetical after-book as an
observed position.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = re.compile(r"^(\d{4}-\d{2}-\d{2})/(\d{6}_[a-fA-F0-9]{6})$")


def _read_object(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _valid_prior_shapes(report: dict) -> bool:
    """Guard every nested prior field consumed by continuity context."""
    final, check, stats = report.get("final"), report.get("final_check"), report.get("stats")
    if not isinstance(final, dict) or not isinstance(check, dict) or not isinstance(stats, dict):
        return False
    if not isinstance(final.get("trades"), list) or any(not isinstance(row, dict) for row in final["trades"]):
        return False
    if not isinstance(check.get("trades"), list) or any(not isinstance(row, dict) for row in check["trades"]):
        return False
    for row in final["trades"]:
        if (row.get("ticker") is not None and not isinstance(row.get("ticker"), str)):
            return False
        if row.get("account_id") is not None and not isinstance(row.get("account_id"), str):
            return False
        for key in ("action", "execution_intent", "condition", "sizing_rationale", "change_reason"):
            if row.get(key) is not None and not isinstance(row.get(key), str):
                return False
        for key in ("shares", "target_shares"):
            value = row.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value)):
                return False
    if final.get("debate_takeaways") is not None and (
            not isinstance(final.get("debate_takeaways"), list)
            or any(not isinstance(row, dict) or any(row.get(key) is not None and not isinstance(row.get(key), str)
                                                   for key in ("analyst", "point", "accepted", "response"))
                   for row in final["debate_takeaways"])):
        return False
    if final.get("changes_vs_draft") is not None and (
            not isinstance(final.get("changes_vs_draft"), list)
            or any(not isinstance(row, dict) or any(row.get(key) is not None and not isinstance(row.get(key), str)
                                                   for key in ("item", "draft", "final", "reason"))
                   for row in final["changes_vs_draft"])):
        return False
    if final.get("cash_decision") is not None and not isinstance(final.get("cash_decision"), dict):
        return False
    cash_decision = final.get("cash_decision")
    if isinstance(cash_decision, dict):
        for key in ("regime", "rationale"):
            if cash_decision.get(key) is not None and not isinstance(cash_decision.get(key), str):
                return False
        for key in ("regime_signals", "deploy_conditions", "raise_conditions"):
            values = cash_decision.get(key)
            if values is not None and (not isinstance(values, list) or any(not isinstance(item, str) for item in values)):
                return False
        target_cash = cash_decision.get("target_cash_pct")
        if target_cash is not None and (isinstance(target_cash, bool) or not isinstance(target_cash, (int, float))
                                        or not math.isfinite(target_cash)):
            return False
    for row in check["trades"]:
        for key in ("ticker", "account_id", "status"):
            if row.get(key) is not None and not isinstance(row.get(key), str):
                return False
        effective = row.get("effective_shares")
        if effective is not None and (isinstance(effective, bool) or not isinstance(effective, (int, float))
                                      or not math.isfinite(effective)):
            return False
    if final.get("decision_summary") is not None and not isinstance(final.get("decision_summary"), str):
        return False
    accounts = stats.get("accounts")
    if accounts is not None and (not isinstance(accounts, list) or any(not isinstance(row, dict) for row in accounts)):
        return False
    if isinstance(accounts, list):
        account_ids = [row.get("account_id") for row in accounts if isinstance(row.get("account_id"), str) and row.get("account_id")]
        if len(account_ids) != len(set(account_ids)):
            return False
        for account in accounts:
            positions = account.get("positions")
            if positions is not None and (not isinstance(positions, list)
                                          or any(not isinstance(row, dict) for row in positions)):
                return False
    if stats.get("report_sources") is not None and not isinstance(stats.get("report_sources"), dict):
        return False
    if isinstance(stats.get("report_sources"), dict) and any(
            not isinstance(row, dict) for row in stats["report_sources"].values()):
        return False
    return True


def _completed_report(path: Path) -> dict | None:
    report = _read_object(path / "portfolio_report.json")
    if not report or not _valid_prior_shapes(report):
        return None
    if report.get("incomplete_stage") or report.get("incomplete_error"):
        return None
    outcome = report.get("review_outcome")
    if outcome is not None and outcome != "complete":
        return None
    check = report.get("final_check")
    if not isinstance(check, dict) or check.get("validated") is not True:
        return None
    if not isinstance(check.get("trades"), list):
        return None
    return report


def _account_snapshots(stats: dict) -> dict[str, dict]:
    """Extract account-local observed quantities and completeness metadata."""
    accounts = stats.get("accounts")
    if not isinstance(accounts, list):
        return {}
    result = {}
    for account in accounts:
        if not isinstance(account, dict):
            continue
        raw_account_id = account.get("account_id")
        account_id = raw_account_id if isinstance(raw_account_id, str) else ""
        if not account_id or account_id in result:
            continue
        raw_positions = account.get("positions")
        positions_complete = isinstance(raw_positions, list)
        quantities = {}
        for position in raw_positions if isinstance(raw_positions, list) else []:
            if not isinstance(position, dict):
                positions_complete = False
                continue
            raw_ticker = position.get("ticker")
            ticker = raw_ticker.strip().upper() if isinstance(raw_ticker, str) else ""
            quantity = position.get("quantity")
            if (not ticker or ticker in quantities or isinstance(quantity, bool)
                    or not isinstance(quantity, (int, float)) or not math.isfinite(quantity) or quantity < 0):
                positions_complete = False
                continue
            quantities[ticker] = float(quantity)
        result[account_id] = {"fetched_at": account.get("fetched_at"),
                              "positions": quantities, "positions_complete": positions_complete}
    return result


def _timestamp(value) -> datetime | None:
    """Parse a full ISO timestamp with explicit timezone and <= microsecond precision."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.(\d+))?(?:Z|[+-]\d{2}:?\d{2})", value)
    if not match or (match.group(1) and len(match.group(1)) > 6):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _snapshot_chronology(prior_value, current_value) -> str:
    old, current = _timestamp(prior_value), _timestamp(current_value)
    if old is None or current is None:
        return "unknown"
    if current > old:
        return "current_newer"
    if current < old:
        return "current_older"
    return "same_instant"


def _observed_quantities(stats: dict) -> dict[tuple[str, str], float]:
    accounts = stats.get("accounts")
    if not isinstance(accounts, list):
        # Aggregate legacy snapshots cannot safely be assigned to a current
        # account, even when today's book happens to contain only one account.
        return {}
    observed: dict[tuple[str, str], float] = {}
    for account in accounts:
        if not isinstance(account, dict):
            continue
        account_id = str(account.get("account_id") or "")
        if not account_id:
            continue
        for position in account.get("positions") or []:
            if not isinstance(position, dict):
                continue
            ticker = str(position.get("ticker") or "").strip().upper()
            quantity = position.get("quantity")
            if (ticker and isinstance(quantity, (int, float)) and not isinstance(quantity, bool)
                    and math.isfinite(quantity) and quantity >= 0):
                observed[(account_id, ticker)] = float(quantity)
    return observed


def previous_review(out_root: Path, as_of: str, current_run_id: str,
                    current_run_started_utc=None) -> tuple[Path, dict] | None:
    """Find latest eligible report strictly earlier than the current run path."""
    try:
        current_date = str(as_of)
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", current_date):
            return None
        if isinstance(current_run_started_utc, datetime):
            current_boundary = (current_run_started_utc.astimezone(timezone.utc)
                                if current_run_started_utc.tzinfo is not None
                                else None)
        else:
            current_boundary = _timestamp(current_run_started_utc)
        candidates = []
        for date_dir in out_root.iterdir():
            if not date_dir.is_dir() or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_dir.name):
                continue
            if date_dir.name > current_date:
                continue
            for run_dir in date_dir.iterdir():
                rel = f"{date_dir.name}/{run_dir.name}"
                if not run_dir.is_dir() or not RUN_DIR.fullmatch(rel):
                    continue
                report = _completed_report(run_dir)
                if not report or report.get("date") != date_dir.name:
                    continue
                if report.get("run_id") not in (None, run_dir.name):
                    continue
                generated = report.get("generated_utc")
                generated_at = _timestamp(generated)
                if current_boundary is not None and generated_at is not None and generated_at > current_boundary:
                    continue
                if date_dir.name == current_date and run_dir.name >= current_run_id:
                    # A precise completion instant can disambiguate repeated
                    # New York wall-clock hours; otherwise use path chronology.
                    if current_boundary is None or generated_at is None or generated_at >= current_boundary:
                        continue
                candidates.append((date_dir.name, run_dir.name, run_dir, report, generated_at))
        if not candidates:
            return None
        precise_ordering = (current_boundary is not None
                            and all(item[4] is not None for item in candidates))
        selected = (max(candidates, key=lambda item: (item[4], item[0], item[1]))
                    if precise_ordering else max(candidates, key=lambda item: (item[0], item[1])))
        _, _, path, report, _ = selected
        report = dict(report)
        report["_continuity_selection_basis"] = "utc_timestamp" if precise_ordering else "path_order_fallback"
        return path, report
    except OSError:
        return None


def build_decision_context(book: dict, prior_path: Path | None, prior: dict | None,
                           current_reports: dict | None = None) -> dict:
    """Compare recorded snapshots and prior advice without asserting execution."""
    if not prior_path or not prior:
        return {"baseline_status": "unavailable", "prior_run_id": None,
                "prior_report_date": None, "prior_decision_summary": None,
                "prior_cash_decision": None, "prior_objections": [],
                "prior_changes_vs_draft": [], "comparisons": [],
                "selection_basis": None, "time_order_uncertainty": "no eligible completed review"}
    prior_stats = prior.get("stats")
    prior_final = prior.get("final")
    prior_check = prior.get("final_check")
    if not _valid_prior_shapes(prior):
        return {"baseline_status": "unavailable", "prior_run_id": None,
                "prior_report_date": None, "prior_decision_summary": None,
                "prior_cash_decision": None, "prior_objections": [],
                "prior_changes_vs_draft": [], "comparisons": [],
                "selection_basis": None, "time_order_uncertainty": "prior report has malformed nested decision data"}

    old_accounts = _account_snapshots(prior_stats)
    current_accounts = _account_snapshots(book)
    old_qty = {(aid, ticker): quantity for aid, snapshot in old_accounts.items()
               for ticker, quantity in snapshot["positions"].items()}
    current_qty = {(aid, ticker): quantity for aid, snapshot in current_accounts.items()
                   for ticker, quantity in snapshot["positions"].items()}

    proposed = {}
    for row in prior_final.get("trades") or []:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        account_id = row.get("account_id")
        if ticker and account_id is not None:
            proposed[(str(account_id), ticker)] = row
    validated = {}
    for row in prior_check.get("trades") or []:
        if isinstance(row, dict) and row.get("ticker") and row.get("account_id"):
            validated[(str(row["account_id"]), str(row["ticker"]).upper())] = row

    keys = set(old_qty) | set(current_qty) | set(proposed)
    comparisons = []
    reports = current_reports or {}
    for account_id, ticker in sorted(keys):
        prior_account = old_accounts.get(account_id)
        current_account = current_accounts.get(account_id)
        old = old_qty.get((account_id, ticker))
        if old is None and prior_account and prior_account["positions_complete"]:
            old = 0.0
        now = current_qty.get((account_id, ticker))
        if now is None and current_account and current_account["positions_complete"]:
            now = 0.0
        prior_row = proposed.get((account_id, ticker)) or {}
        checked = validated.get((account_id, ticker)) or {}
        report = reports.get(ticker) or {}
        old_source = (prior_stats.get("report_sources") or {}).get(ticker) or {}
        old_report_dir = old_source.get("report_dir")
        current_report_dir = report.get("report_dir")
        evidence_change = ("unknown" if not old_report_dir or not current_report_dir else
                           "same_report" if old_report_dir == current_report_dir else "different_report")
        prior_fetched_at = prior_account.get("fetched_at") if prior_account else None
        current_fetched_at = current_account.get("fetched_at") if current_account else None
        chronology = _snapshot_chronology(prior_fetched_at, current_fetched_at)
        delta = (now - old if old is not None and now is not None and chronology == "current_newer"
                 else None)
        if chronology == "current_newer" and old is not None and now is not None:
            comparison_status = "comparable_current_newer"
        elif chronology == "current_older":
            comparison_status = "current_snapshot_older"
        elif chronology == "same_instant":
            comparison_status = "same_snapshot_time"
        elif prior_account is None or current_account is None:
            comparison_status = "account_unavailable"
        elif old is None or now is None:
            comparison_status = "position_snapshot_incomplete"
        else:
            comparison_status = "snapshot_time_unknown"
        comparisons.append({
            "account_id": account_id,
            "ticker": ticker,
            "observed_shares": now,
            "prior_observed_shares": old,
            "quantity_change": delta,
            "prior_snapshot_fetched_at": prior_fetched_at,
            "current_snapshot_fetched_at": current_fetched_at,
            "snapshot_chronology": chronology,
            "prior_decision": prior_row.get("action"),
            "prior_proposed_shares": prior_row.get("shares"),
            "prior_target_shares": prior_row.get("target_shares"),
            "prior_intent": prior_row.get("execution_intent"),
            "prior_condition": prior_row.get("condition"),
            "prior_sizing_rationale": prior_row.get("sizing_rationale"),
            "prior_change_reason": prior_row.get("change_reason"),
            "prior_validation_status": checked.get("status"),
            "prior_effective_hypothetical_shares": checked.get("effective_shares"),
            "prior_research_status": old_source.get("source_profile", "unknown"),
            "prior_research_date": old_source.get("source_date"),
            "research_evidence_change": evidence_change,
            "current_research_status": report.get("source_status", "missing"),
            "current_research_date": report.get("analysis_date"),
            "comparison_status": comparison_status,
            "execution_attribution": "unknown; observed quantity changes do not prove fills",
        })
    return {
        "baseline_status": "available",
        "prior_run_id": prior_path.name,
        "prior_report_date": prior.get("date"),
        "selection_basis": prior.get("_continuity_selection_basis", "unknown"),
        "time_order_uncertainty": (None if prior.get("_continuity_selection_basis") == "utc_timestamp"
                                   else "prior selected by New York report path order because precise timestamp provenance was unavailable"),
        "prior_decision_summary": str(prior_final.get("decision_summary") or "")[:2000],
        "prior_cash_decision": prior_final.get("cash_decision"),
        "prior_objections": [
            {"analyst": str(row.get("analyst") or "")[:100],
             "point": str(row.get("point") or "")[:800],
             "accepted": row.get("accepted"),
             "response": str(row.get("response") or "")[:800]}
            for row in (prior_final.get("debate_takeaways") or [])[:20] if isinstance(row, dict)
        ],
        "prior_changes_vs_draft": [
            {"item": str(row.get("item") or "")[:200],
             "draft": str(row.get("draft") or "")[:300],
             "final": str(row.get("final") or "")[:300],
             "reason": str(row.get("reason") or "")[:800]}
            for row in (prior_final.get("changes_vs_draft") or [])[:20] if isinstance(row, dict)
        ],
        "comparisons": comparisons,
    }


def attach_position_results(decision: dict, stats: dict, validation: dict) -> list[dict]:
    """Bind requested intent to observed and hypothetical quantities for review."""
    observed = _observed_quantities(stats)
    checked = {(str(row.get("account_id") or ""), str(row.get("ticker") or "").upper()): row
               for row in validation.get("trades") or [] if isinstance(row, dict)}
    results = []
    for proposal in decision.get("trades") or []:
        if not isinstance(proposal, dict):
            continue
        account_id = str(proposal.get("account_id") or "")
        ticker = str(proposal.get("ticker") or "").strip().upper()
        row = checked.get((account_id, ticker), {})
        current = observed.get((account_id, ticker), 0.0)
        effective = float(row.get("effective_shares") or 0.0)
        action = str(proposal.get("action") or "HOLD").upper()
        after = current + effective if action == "BUY" else current - effective if action == "SELL" else current
        target = proposal.get("target_shares")
        gap = None
        status = "unknown_target"
        if isinstance(target, (int, float)) and not isinstance(target, bool) and math.isfinite(target) and target >= 0:
            gap = float(target) - after
            status = "reached" if abs(gap) < 1e-9 else "above_target" if gap < 0 else "remaining"
        if row.get("status") in {"rejected", "adjusted"}:
            status = "validation_conflict"
        elif proposal.get("execution_intent") == "deferred" or row.get("status") == "deferred":
            status = "deferred"
        results.append({"account_id": account_id, "ticker": ticker,
                        "observed_shares": current,
                        "hypothetical_after_shares": after,
                        "target_shares": target,
                        "target_gap_shares": gap,
                        "position_status": status,
                        "validation_status": row.get("status")})
    return results


def validate_decision_contract(decision: dict, stats: dict) -> None:
    """Reject omitted/contradictory coverage while leaving sizing judgment free."""
    if not isinstance(decision, dict) or decision.get("decision_contract_version") != 2:
        raise ValueError("Fund manager decision contract version 2 is required")
    accounts = stats.get("accounts") or []
    if not accounts:
        raise ValueError("Fund manager account context is unavailable")
    holding_quantities = {(str(account.get("account_id") or ""), str(pos.get("ticker") or "").upper()):
                          float(pos.get("quantity") or 0)
                          for account in accounts for pos in account.get("positions") or []}
    holdings = set(holding_quantities)
    rows = decision.get("trades")
    if not isinstance(rows, list):
        raise ValueError("trades must be a list")
    identities = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each trade must be an object")
        account = str(row.get("account_id") or "")
        ticker = str(row.get("ticker") or "").strip().upper()
        identity = (account, ticker)
        identities.append(identity)
        if not account or not ticker:
            raise ValueError("every trade must identify an account and ticker")
        intent = row.get("execution_intent")
        action = str(row.get("action") or "").upper()
        shares = row.get("shares")
        if intent not in {"current", "deferred", "none"}:
            raise ValueError(f"{ticker}: invalid execution intent")
        if isinstance(shares, bool) or not isinstance(shares, (int, float)) or not math.isfinite(shares) or shares < 0:
            raise ValueError(f"{ticker}: shares must be a finite nonnegative number")
        if action == "HOLD":
            if intent != "none" or shares != 0:
                raise ValueError(f"{ticker}: HOLD requires no execution intent and zero shares")
        elif action in {"BUY", "SELL"}:
            if intent not in {"current", "deferred"} or shares <= 0:
                raise ValueError(f"{ticker}: BUY/SELL requires a positive current or deferred quantity")
            if intent == "deferred" and not str(row.get("condition") or "").strip():
                raise ValueError(f"{ticker}: deferred intent requires a future condition")
        else:
            raise ValueError(f"{ticker}: unsupported action")
        if not str(row.get("review_horizon") or "").strip():
            raise ValueError(f"{ticker}: review horizon must be explicit")
        if not str(row.get("sizing_rationale") or "").strip():
            raise ValueError(f"{ticker}: sizing rationale must be explicit")
        if not str(row.get("change_reason") or "").strip():
            raise ValueError(f"{ticker}: prior-decision continuity reason must be explicit")
        target = row.get("target_shares")
        if target is not None and (isinstance(target, bool) or not isinstance(target, (int, float))
                                   or not math.isfinite(target) or target < 0):
            raise ValueError(f"{ticker}: target_shares must be null or a finite nonnegative number")
        if target is not None:
            current = holding_quantities.get(identity, 0.0)
            if action == "BUY" and target <= current:
                raise ValueError(f"{ticker}: BUY target must be above observed shares")
            if action == "SELL" and target >= current:
                raise ValueError(f"{ticker}: SELL target must be below observed shares")
            if action == "HOLD" and abs(target - current) > 1e-9:
                raise ValueError(f"{ticker}: HOLD target must match observed shares")
            if action == "BUY" and shares > target - current + 1e-9:
                raise ValueError(f"{ticker}: proposed buy exceeds the stated target exposure")
            if action == "SELL" and shares > current - target + 1e-9:
                raise ValueError(f"{ticker}: proposed sale exceeds the stated target reduction")
    if len(identities) != len(set(identities)):
        raise ValueError("each account/ticker may appear only once in the plan")
    if not holdings.issubset(set(identities)):
        omitted = sorted(holdings - set(identities))
        raise ValueError("every held account/ticker requires a decision: " + ", ".join(f"{a}/{t}" for a, t in omitted))

    expected_candidates = {str(p.get("ticker") or "").upper() for p in stats.get("watchlist") or []}
    coverage = decision.get("candidate_coverage")
    if not isinstance(coverage, list):
        raise ValueError("candidate_coverage must be a list")
    coverage_tickers = [str(item.get("ticker") or "").strip().upper() for item in coverage if isinstance(item, dict)]
    if len(coverage_tickers) != len(coverage) or len(coverage_tickers) != len(set(coverage_tickers)):
        raise ValueError("candidate_coverage must contain unique ticker objects")
    if set(coverage_tickers) != expected_candidates:
        raise ValueError("candidate_coverage must disposition every watchlist ticker exactly once")
    buy_tickers = {str(row.get("ticker") or "").upper() for row in rows if str(row.get("action") or "").upper() == "BUY"}
    coverage_by_ticker = {str(item["ticker"]).upper(): item for item in coverage}
    for ticker, item in coverage_by_ticker.items():
        disposition = item.get("disposition")
        if disposition not in {"selected", "watch", "pass", "insufficient_evidence"}:
            raise ValueError(f"{ticker}: invalid candidate disposition")
        if not str(item.get("rationale") or "").strip():
            raise ValueError(f"{ticker}: candidate disposition requires a reason")
        source = (stats.get("report_sources") or {}).get(ticker) or {}
        expected_status = source.get("source_profile", "missing")
        expected_date = source.get("analysis_date")
        if item.get("source_profile") != expected_status:
            raise ValueError(f"{ticker}: candidate source_profile does not match saved evidence")
        if item.get("research_status") != source.get("status", "missing"):
            raise ValueError(f"{ticker}: candidate research_status does not match saved evidence freshness")
        if item.get("evidence_date") != expected_date:
            raise ValueError(f"{ticker}: candidate evidence_date does not match saved evidence")
        if (disposition == "selected") != (ticker in buy_tickers):
            raise ValueError(f"{ticker}: selected status must match an account-scoped BUY")
