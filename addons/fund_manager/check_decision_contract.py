"""Offline fixtures for fund decision continuity, source boundaries, and v2 contracts."""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fund_manager as fm
from decision_continuity import build_decision_context, previous_review, validate_decision_contract


def _decision(action="HOLD", target=10):
    return {
        "decision_contract_version": 2,
        "trades": [{"account_id": "acct", "ticker": "AAA", "action": action,
                    "decision": "keep" if action == "HOLD" else "add" if action == "BUY" else "trim",
                    "shares": 0 if action == "HOLD" else 2, "execution_intent": "none" if action == "HOLD" else "current",
                    "target_shares": target, "review_horizon": "weekly review",
                    "sizing_rationale": "Based on total portfolio risk and alternatives.",
                    "change_reason": "No prior eligible change."}],
        "candidate_coverage": [{"ticker": "NEW", "disposition": "watch", "rationale": "Evidence is incomplete.",
                                "research_status": "missing", "source_profile": "missing", "evidence_date": None}],
    }


def _check_nov14_to_latest_nov17():
    """Exercise the accepted NVDA/XLV/XLP sequence through continuity and validation."""
    names = ("NVDA", "XLV", "XLP")
    old_reports = {ticker: {"report_dir": f"{ticker}_20261114_090000",
                            "source_profile": "legacy_portfolio_aware",
                            "source_date": "2026-11-14", "analysis_date": "2026-11-14",
                            "status": "current"}
                   for ticker in names}
    prior_trades = [
        {"ticker": "NVDA", "account_id": "acct", "is_holding": True,
         "investment_role": "swing", "decision": "trim", "action": "SELL", "shares": 10,
         "est_price": 200, "order_type": "market", "limit_price": None, "timing": "current",
         "agent_rating": "legacy excluded", "follows_agent": "overrides", "rationale": "Reduce event risk."},
        {"ticker": "XLV", "account_id": "acct", "is_holding": True,
         "investment_role": "long_term", "decision": "keep", "action": "HOLD", "shares": 0,
         "est_price": 160, "order_type": "none", "limit_price": None, "timing": "review weekly",
         "agent_rating": "legacy excluded", "follows_agent": "sizes_differently",
         "rationale": "Hold 23 shares; consider adding up to six only on a pullback. No immediate order."},
        {"ticker": "XLP", "account_id": "acct", "is_holding": False,
         "investment_role": "swing", "decision": "new_position", "action": "BUY", "shares": 12,
         "est_price": 100, "order_type": "market", "limit_price": None, "timing": "current",
         "agent_rating": "legacy excluded", "follows_agent": "no_agent_report", "rationale": "Start diversified exposure."},
    ]
    prior_report = {
        "date": "2026-11-14", "run_id": "140000_a14a14", "generated_utc": "2026-11-14T19:00:00Z",
        "review_outcome": "complete",
        "stats": {
            "accounts": [{"account_id": "acct", "fetched_at": "2026-11-14T19:00:00Z", "cash": 5000,
                           "positions": [{"ticker": "NVDA", "quantity": 35},
                                         {"ticker": "XLV", "quantity": 23}]}],
            "report_sources": old_reports,
        },
        "final": {
            "trades": prior_trades,
            "decision_summary": "Reduce NVDA risk, retain XLV, and initiate XLP exposure.",
            "cash_decision": {"target_cash_pct": 30, "regime": "neutral", "regime_signals": [],
                              "rationale": "Keep flexibility while comparing candidates.",
                              "deploy_conditions": [], "raise_conditions": []},
            "debate_takeaways": [
                {"analyst": "conservative", "point": "NVDA event risk favors a smaller exposure.",
                 "accepted": "accepted", "response": "Trim ten shares and review again."},
                {"analyst": "aggressive", "point": "Avoid over-trimming NVDA into weakness.",
                 "accepted": "partly", "response": "Keep a meaningful residual position."},
                {"analyst": "conservative", "point": "XLP adds defensive crowding with weak cash-funded volatility benefit.",
                 "accepted": "accepted", "response": "Cut the draft buy from 25 to 12 shares and retain entry conditions."},
            ],
            "changes_vs_draft": [
                {"item": "NVDA", "draft": "SELL 6", "final": "SELL 10",
                 "reason": "Risk committee event-risk objection."},
                {"item": "XLP", "draft": "BUY 25", "final": "BUY 12",
                 "reason": "Defensive crowding and weak cash-funded volatility benefit; preserve entry conditions."},
                {"item": "XLV", "draft": "HOLD with conditional add up to 6", "final": "HOLD 0",
                 "reason": "No pullback trigger was present in the prior report."},
            ],
        },
        "final_check": {"validated": True, "trades": [
            {"ticker": "NVDA", "account_id": "acct", "status": "accepted", "effective_shares": 10},
            {"ticker": "XLV", "account_id": "acct", "status": "hold", "effective_shares": 0},
            {"ticker": "XLP", "account_id": "acct", "status": "accepted", "effective_shares": 12},
        ], "accounts_after": [{"account_id": "acct", "positions": [
            {"ticker": "NVDA", "quantity": 25}, {"ticker": "XLV", "quantity": 23},
            {"ticker": "XLP", "quantity": 12}]}]},
    }

    with tempfile.TemporaryDirectory(prefix="fm_nov14_nov17_") as temp:
        out_root = Path(temp)
        prior_dir = out_root / "2026-11-14" / "140000_a14a14"
        prior_dir.mkdir(parents=True)
        (prior_dir / "portfolio_report.json").write_text(json.dumps(prior_report))
        chosen = previous_review(out_root, "2026-11-17", "170000_b17b17", "2026-11-17T22:00:00Z")
        assert chosen and chosen[0] == prior_dir

        current_reports = {ticker: {"ticker": ticker, "report_dir": f"{ticker}_20261114_090000",
                                    "analysis_date": "2026-11-14", "generated_utc": "2026-11-14T19:00:00Z",
                                    "source_status": "legacy_portfolio_aware", "legacy_evidence": {}}
                           for ticker in names}
        prices = {"NVDA": 200.0, "XLV": 160.0, "XLP": 100.0}
        quantities = {"NVDA": 25.0, "XLV": 23.0, "XLP": 12.0}
        current_positions = [{"ticker": ticker, "quantity": quantities[ticker], "price": prices[ticker],
                              "average_price": prices[ticker] - 10, "sellable_quantity": quantities[ticker],
                              "theme": "defensive", "type": "ETF" if ticker != "NVDA" else "EQUITY",
                              "stress_beta": 1.0, "beta_1y": 1.0}
                             for ticker in names]
        current_account = {"account_id": "acct", "label": "Fixture account", "broker": "fixture",
                           "currency": "USD", "cash": 5000.0, "available_cash": 5000.0,
                           "fetched_at": "2026-11-17T21:00:00Z",
                           "positions": [{k: v for k, v in row.items() if k not in ("price", "theme", "type", "stress_beta", "beta_1y")}
                                         for row in current_positions]}
        book_positions = [{"ticker": ticker, "quantity": quantities[ticker], "price": prices[ticker],
                           "average_price": prices[ticker] - 10, "theme": "defensive",
                           "type": "ETF" if ticker != "NVDA" else "EQUITY",
                           "stress_beta": 1.0, "beta_1y": 1.0}
                          for ticker in names]
        book = {"cash": 5000.0, "positions": book_positions, "accounts": [current_account]}
        freshness = {ticker: fm.source_freshness(current_reports[ticker], "2026-11-17") for ticker in names}
        stats = {
            "accounts": [current_account], "positions": current_positions, "watchlist": [],
            "report_sources": {ticker: {**freshness[ticker], "report_dir": current_reports[ticker]["report_dir"],
                                         "source_profile": "legacy_portfolio_aware", "analysis_date": "2026-11-14"}
                               for ticker in names},
            "account": {"cash": 5000.0, "total_value": 14880.0, "cash_pct": 33.6},
            "cash_policy": {"min_pct": 0, "max_pct": 100, "soft_default_pct": 30},
            "risk_policy": dict(fm.RISK_POLICY), "trade_policy": fm.parse_trade_policy({}),
        }
        decision_context = build_decision_context(book, *chosen, current_reports)
        assert decision_context["prior_report_date"] == "2026-11-14"
        assert any("event risk" in row["point"].lower() for row in decision_context["prior_objections"])
        assert any("defensive crowding" in row["point"].lower() for row in decision_context["prior_objections"])
        assert decision_context["prior_changes_vs_draft"][0]["reason"] == "Risk committee event-risk objection."
        prior_xlp_change = next(row for row in decision_context["prior_changes_vs_draft"] if row["item"] == "XLP")
        assert prior_xlp_change["draft"] == "BUY 25" and prior_xlp_change["final"] == "BUY 12"
        assert "defensive crowding" in prior_xlp_change["reason"].lower()
        prior_xlv_change = next(row for row in decision_context["prior_changes_vs_draft"] if row["item"] == "XLV")
        assert "conditional add up to 6" in prior_xlv_change["draft"]
        comparisons = {(row["account_id"], row["ticker"]): row for row in decision_context["comparisons"]}
        assert (comparisons[("acct", "NVDA")]["prior_observed_shares"],
                comparisons[("acct", "NVDA")]["observed_shares"],
                comparisons[("acct", "NVDA")]["quantity_change"]) == (35, 25, -10)
        assert (comparisons[("acct", "XLV")]["prior_observed_shares"],
                comparisons[("acct", "XLV")]["observed_shares"],
                comparisons[("acct", "XLV")]["quantity_change"]) == (23, 23, 0)
        assert (comparisons[("acct", "XLP")]["prior_observed_shares"],
                comparisons[("acct", "XLP")]["observed_shares"],
                comparisons[("acct", "XLP")]["quantity_change"]) == (0, 12, 12)
        assert comparisons[("acct", "NVDA")]["prior_proposed_shares"] == 10
        assert comparisons[("acct", "NVDA")]["prior_target_shares"] is None
        assert comparisons[("acct", "NVDA")]["prior_effective_hypothetical_shares"] == 10
        assert comparisons[("acct", "XLP")]["prior_proposed_shares"] == 12
        assert comparisons[("acct", "XLP")]["prior_target_shares"] is None
        assert comparisons[("acct", "XLV")]["prior_target_shares"] is None
        assert "prior_hypothetical_after_shares" not in comparisons[("acct", "NVDA")]
        assert "decision_contract_version" not in prior_report["final"]
        assert all(freshness[ticker]["status"] == "reused" for ticker in names)
        assert all(stats["report_sources"][ticker]["status"] == "reused" for ticker in names)

        context = fm.build_context(book, {}, stats, current_reports, None, "2026-11-17",
                                   decision_context=decision_context)
        assert "2026-11-14" in context and "conservative" in context and "do not prove fills" in context
        assert "NVDA" in context and "XLV" in context and "XLP" in context
        assert '"prior_proposed_shares": 10' in context and '"prior_target_shares": null' in context
        assert "Risk committee event-risk objection." in context
        assert "accounts_after" not in context
        assert "Defensive crowding and weak cash-funded volatility benefit" in context
        assert "HOLD with conditional add up to 6" in context

        trades = [
            {"ticker": "NVDA", "account_id": "acct", "is_holding": True, "investment_role": "swing",
             "decision": "trim", "action": "SELL", "shares": 6, "est_price": 200.0,
             "order_type": "market", "limit_price": None, "timing": "current review",
             "agent_rating": "legacy evidence reused", "follows_agent": "sizes_differently",
             "rationale": "Reassess event risk against the full book.", "target_shares": 19,
             "review_horizon": "weekly review", "sizing_rationale": "Reduce concentrated risk while retaining exposure.",
             "execution_intent": "current", "condition": "Execute at the current supplied quote.",
             "change_reason": "Revisit the prior event-risk trim using the newer account snapshot."},
            {"ticker": "XLV", "account_id": "acct", "is_holding": True, "investment_role": "long_term",
             "decision": "add", "action": "BUY", "shares": 8, "est_price": 160.0,
             "order_type": "market", "limit_price": None, "timing": "current review",
             "agent_rating": "legacy evidence reused", "follows_agent": "sizes_differently",
             "rationale": "Compare current opportunity with retained holdings.", "target_shares": 31,
             "review_horizon": "monthly review", "sizing_rationale": "Incremental exposure fits portfolio risk.",
             "execution_intent": "current", "condition": "Execute at the current supplied quote.",
             "change_reason": "Revisit the prior conditional six-share idea in light of current evidence."},
            {"ticker": "XLP", "account_id": "acct", "is_holding": True, "investment_role": "swing",
             "decision": "add", "action": "BUY", "shares": 12, "est_price": 100.0,
             "order_type": "market", "limit_price": None, "timing": "current review",
             "agent_rating": "legacy evidence reused", "follows_agent": "sizes_differently",
             "rationale": "Reassess the previous starter against the current opportunity set.", "target_shares": 24,
             "review_horizon": "weekly review", "sizing_rationale": "Compare staged entry with other uses of capital.",
             "execution_intent": "current", "condition": "Use the current supplied quote after reviewing entry evidence.",
             "change_reason": "Revisit the prior 25-share draft cut to 12 and reassess the destination with current holdings."},
        ]
        final = {
            "decision_contract_version": 2, "headline": "Reassess the whole-book targets.",
            "diagnosis": "Use dated research and current account snapshots.",
            "cash_decision": {"target_cash_pct": 30, "regime": "neutral", "regime_signals": [],
                              "rationale": "Maintain flexibility.", "deploy_conditions": [], "raise_conditions": []},
            "key_risks": [], "correlation_clusters": [], "decision_summary": "Review three exposures.",
            "trades": trades, "candidate_coverage": [], "swaps_considered": [], "priority_order": list(names),
            "cash_plan": {"summary": "No assumed future proceeds.", "deployments": []}, "levels": [],
            "what_would_change_my_mind": [], "data_gaps": [], "confidence": "medium",
            "debate_takeaways": [], "changes_vs_draft": [],
        }
        fm._validate_schema(final, fm.FINAL_SCHEMA)
        check = fm.validate_trades(final, stats)
        final["decision_context"] = decision_context
        final["position_results"] = fm.attach_position_results(final, stats, check)
        assert check["validated"] is True
        rows = {(row["account_id"], row["ticker"]): row for row in check["trades"]}
        assert rows[("acct", "NVDA")]["status"] == "accepted" and rows[("acct", "NVDA")]["effective_shares"] == 6
        assert rows[("acct", "XLV")]["status"] == "accepted" and rows[("acct", "XLV")]["effective_shares"] == 8
        assert rows[("acct", "XLP")]["status"] == "accepted" and rows[("acct", "XLP")]["effective_shares"] == 12
        assert check["ending_cash"] == 3711
        results = {(row["account_id"], row["ticker"]): row for row in final["position_results"]}
        assert (results[("acct", "NVDA")]["observed_shares"],
                results[("acct", "NVDA")]["hypothetical_after_shares"],
                results[("acct", "NVDA")]["target_shares"]) == (25, 19.0, 19)
        assert (results[("acct", "XLV")]["observed_shares"],
                results[("acct", "XLV")]["hypothetical_after_shares"],
                results[("acct", "XLV")]["target_shares"]) == (23, 31.0, 31)
        assert (results[("acct", "XLP")]["observed_shares"],
                results[("acct", "XLP")]["hypothetical_after_shares"],
                results[("acct", "XLP")]["target_shares"],
                results[("acct", "XLP")]["target_gap_shares"],
                results[("acct", "XLP")]["position_status"]) == (12, 24.0, 24, 0.0, "reached")
        assert all(row["change_reason"] and row["review_horizon"] and row["condition"]
                   for row in final["trades"])


def main():
    stats = {"accounts": [{"account_id": "acct", "positions": [{"ticker": "AAA", "quantity": 10}]}],
             "positions": [{"ticker": "AAA"}], "watchlist": [{"ticker": "NEW"}],
             "report_sources": {"NEW": {"source_profile": "missing", "status": "missing", "analysis_date": None}}}
    validate_decision_contract(_decision(), stats)
    hold_with_deferred_intent = _decision()
    hold_with_deferred_intent["trades"][0]["execution_intent"] = "deferred"
    original_hold = json.loads(json.dumps(hold_with_deferred_intent))
    try:
        validate_decision_contract(hold_with_deferred_intent, stats)
        raise AssertionError("HOLD with deferred intent was accepted")
    except ValueError as exc:
        assert "HOLD requires no execution intent and zero shares" in str(exc)
    assert hold_with_deferred_intent == original_hold, "validation must not rewrite contradictory model output"

    deferred_buy = _decision("BUY", target=12)
    deferred_buy["trades"][0].update(shares=2, execution_intent="deferred", condition="Only after support holds.")
    validate_decision_contract(deferred_buy, stats)
    deferred_sell = _decision("SELL", target=8)
    deferred_sell["trades"][0].update(shares=2, execution_intent="deferred", condition="If support breaks.")
    validate_decision_contract(deferred_sell, stats)

    runtime_contract = "## Decision encoding contract (applies to draft and final plans)"
    fixture_book = {"cash": 1000, "positions": [{"ticker": "AAA"}]}
    minimal_context = fm.build_context(
        fixture_book, {"portfolio_guidance": {"cash_reference_pct": 10,
                                               "long_term_pct": 30, "swing_pct": 60}},
        {"accounts": stats["accounts"], "positions": stats["positions"], "watchlist": stats["watchlist"]},
        {}, None, "2026-11-17")
    # Manager system prompts may be replaced by saved custom prompts. This
    # runtime-owned context is passed separately to both draft and final calls.
    assert runtime_contract in minimal_context
    assert "HOLD as shares=0 and execution_intent=none" in minimal_context
    assert "Set target_shares to the observed account quantity" in minimal_context
    assert "Use execution_intent=deferred only for an actual conditional BUY or SELL" in minimal_context
    assert "null is allowed when unavailable" in minimal_context
    book_section = minimal_context.split("## Book (shares, average cost, cash)\n", 1)[1].split(
        "\n\n## Owner-approved allocation targets", 1)[0]
    assert json.loads(book_section) == fixture_book
    owner_section = minimal_context.split("## Owner-approved allocation targets", 1)[1].split(
        "\n\n## Trade policy", 1)[0]
    assert '"portfolio_guidance"' in owner_section
    assert "## Owner-approved portfolio guidelines" in owner_section
    assert "Use 10% cash, 30% long-term and 60% swing" in owner_section
    assert fm.TRADE_ITEM["properties"]["shares"]["description"].startswith("Proposed order quantity")
    assert "proposed total target position/exposure" in fm.TRADE_ITEM["properties"]["target_shares"]["description"].lower()
    assert "staged order may move partway toward that target" in fm.TRADE_ITEM["properties"]["target_shares"]["description"]
    assert "HOLD requires none" in fm.TRADE_ITEM["properties"]["execution_intent"]["description"]
    assert "reassessment note does not imply an order" in fm.TRADE_ITEM["properties"]["condition"]["description"]
    assert "execution_intent" not in fm.LEGACY_TRADE_ITEM["properties"]
    # Candidate prompt metadata is the exact producer shape consumed by the
    # current validator, for both new and legacy source profiles.
    coverage_marker = "## Required candidate coverage universe (include every row exactly once in candidate_coverage)"
    for profile, day in (("research_only", "2026-11-17"), ("legacy_portfolio_aware", "2026-11-14")):
        source_report = {"ticker": "NEW", "report_dir": "NEW_" + day.replace("-", "") + "_120000",
                         "analysis_date": day, "generated_utc": day + "T12:00:00Z",
                         "source_status": profile,
                         "research_brief": ({"outlook": "neutral", "thesis": "Evidence.",
                                              "counter_thesis": "Risks.", "valuation": {},
                                              "sentiment": {"summary": "Mixed"}}
                                             if profile == "research_only" else None),
                         "legacy_evidence": None}
        source_status = "current" if profile == "research_only" else "reused"
        context_stats = {"accounts": stats["accounts"], "positions": stats["positions"],
                         "watchlist": stats["watchlist"], "cash_policy": {},
                         "report_sources": {"NEW": {"source_profile": profile, "status": source_status,
                                                      "analysis_date": day, "source_date": day}}}
        context = fm.build_context({"cash": 10, "positions": [{"ticker": "AAA"}]}, {}, context_stats,
                                   {"NEW": source_report}, None, "2026-11-17")
        coverage_json = context.split(coverage_marker + "\n\n", 1)[1].split("\n\n", 1)[0]
        supplied = json.loads(coverage_json)
        assert supplied == [{"ticker": "NEW", "research_status": source_status,
                             "source_profile": profile, "evidence_date": day}], supplied
        candidate_decision = _decision()
        candidate_decision["candidate_coverage"] = [{**supplied[0], "disposition": "watch",
                                                       "rationale": "Wait for more evidence."}]
        validate_decision_contract(candidate_decision, context_stats)
    for bad in (_decision("BUY", 10), _decision("SELL", 10)):
        try:
            validate_decision_contract(bad, stats)
            raise AssertionError("action/target mismatch was accepted")
        except ValueError:
            pass
    bad_coverage = _decision()
    bad_coverage["candidate_coverage"][0]["source_profile"] = "research_only"
    try:
        validate_decision_contract(bad_coverage, stats)
        raise AssertionError("invented coverage provenance was accepted")
    except ValueError:
        pass

    # New research brief is allowlisted; malformed profiles are unknown, while
    # old manager/trader sizing prose never reaches the portfolio prompt.
    with tempfile.TemporaryDirectory(prefix="fm_contract_") as temp:
        root = Path(temp)
        report_dir = root / "AAA_20261117_120000"
        (report_dir / "5_portfolio").mkdir(parents=True)
        (report_dir / "3_trading").mkdir(parents=True)
        (report_dir / "2_research").mkdir(parents=True)
        (report_dir / "1_analysts").mkdir(parents=True)
        (report_dir / "complete_report.md").write_text("Analysis date: 2026-11-17\nGenerated: 2026-11-17T12:00:00Z\n")
        (report_dir / "5_portfolio/decision.md").write_text("PM says buy 50 shares; allocate 20% of account.\n")
        (report_dir / "3_trading/trader.md").write_text("**Position Sizing**: 50 shares for this account.\n")
        (report_dir / "2_research/manager.md").write_text("Use 15% of account.\n")
        (report_dir / "1_analysts/market.md").write_text("Instrument trend evidence only.\n")
        (report_dir / "2_research/bull.md").write_text("## Upside\nStrong product cycle.\n")
        (report_dir / "2_research/bear.md").write_text("## Downside\nHigh valuation risk.\n")
        parsed = fm.parse_report("AAA", report_dir, include_risk=True, max_chars=4000)
        context = fm.build_context({"cash": 10, "positions": [{"ticker": "AAA"}]}, {},
                                   {"positions": [], "watchlist": [], "cash_policy": {}},
                                   {"AAA": parsed}, None, "2026-11-17")
        assert "buy 50 shares" not in context and "50 shares for this account" not in context
        assert "15% of account" not in context and "Instrument trend evidence only" in context
        assert "legacy_portfolio_aware" in context

        brief = {"schema_version": 1, "profile": "research_only", "ticker": "AAA",
                 "analysis_date": "2026-11-17", "generated_utc": "2026-11-17T12:00:00Z",
                 "outlook": "attractive", "thesis": "Durable demand supports growth.",
                 "counter_thesis": "Valuation leaves little room for error.",
                 "valuation": {"target_price": 210, "target_range": {"low": 190, "high": 225},
                               "method": "DCF", "assumptions": "Revenue growth moderates."},
                 "thesis_horizon": "12 to 18 months", "catalysts": ["Quarterly results"],
                 "risks": ["Multiple compression"], "sentiment": {"summary": "Mixed", "as_of": "2026-11-17"},
                 "conditional_levels": [{"condition": "holds support", "level": 180, "meaning": "confirmation"}],
                 "confidence": "medium", "unknowns": ["Demand durability"],
                 "evidence_dates": [{"source": "Quarterly filing", "as_of": "2026-10-28"}]}
        (report_dir / "research_brief.json").write_text(json.dumps(brief))
        parsed_brief = fm.parse_report("AAA", report_dir, include_risk=False, max_chars=4000)
        brief_context = fm.build_context({"cash": 10, "positions": [{"ticker": "AAA"}]}, {},
                                         {"positions": [], "watchlist": [], "cash_policy": {}},
                                         {"AAA": parsed_brief}, None, "2026-11-17")
        assert parsed_brief["source_status"] == "research_only"
        assert parsed_brief["research_brief"]["valuation"]["target_range"] == {"low": 190, "high": 225}
        assert "12 to 18 months" in brief_context and "buy 50 shares" not in brief_context

        unsafe_brief = {**brief, "position_sizing": "buy 20 shares"}
        (report_dir / "research_brief.json").write_text(json.dumps(unsafe_brief))
        malformed = fm.parse_report("AAA", report_dir, include_risk=False, max_chars=4000)
        assert malformed["source_status"] == "unknown" and malformed["legacy_evidence"] is None

    # Select the latest eligible completed review by its run path, excluding
    # snapshots, incomplete reports, raw/recovery artifacts, and future paths.
    with tempfile.TemporaryDirectory(prefix="fm_prior_") as temp:
        root = Path(temp)
        def save(day, run, *, complete=True, snapshot=False, generated=None):
            folder = root / day / run
            folder.mkdir(parents=True)
            report = {"date": day, "run_id": run, "review_outcome": "complete" if complete else "incomplete",
                      "generated_utc": generated or (day + "T12:00:00Z"),
                      "final": {"trades": [{"ticker": "AAA", "account_id": "acct", "action": "SELL",
                                                "shares": 2, "target_shares": 8, "execution_intent": "current",
                                                "sizing_rationale": "Trim risk", "change_reason": "Event risk"}]},
                      "final_check": {"validated": True, "trades": [{"ticker": "AAA", "account_id": "acct",
                                                                          "status": "accepted", "effective_shares": 2}]},
                      "stats": {"accounts": [{"account_id": "acct", "fetched_at": day + "T12:00:00Z",
                                                 "positions": [{"ticker": "AAA", "quantity": 10}]}]}}
            (folder / ("portfolio_snapshot.json" if snapshot else "portfolio_report.json")).write_text(json.dumps(report))
            return folder
        save("2026-11-16", "090000_111111")
        latest = save("2026-11-17", "100000_222222")
        save("2026-11-17", "110000_333333", complete=False)
        save("2026-11-17", "120000_444444", snapshot=True)
        save("2026-11-17", "123000_444445", generated="2026-11-18T00:00:00Z")
        malformed = save("2026-11-17", "124000_444446")
        malformed_report_path = malformed / "portfolio_report.json"
        malformed_report = json.loads(malformed_report_path.read_text())
        malformed_report["final"]["debate_takeaways"] = {"unexpected": "object"}
        malformed_report_path.write_text(json.dumps(malformed_report))
        save("2026-11-18", "080000_555555")
        selected = previous_review(root, "2026-11-17", "130000_aaaaaa", "2026-11-17T18:00:00Z")
        assert selected and selected[0] == latest
        current_book = {"accounts": [{"account_id": "acct", "fetched_at": "2026-11-17T13:00:00Z",
                                       "positions": [{"ticker": "AAA", "quantity": 7}]}]}
        continuity = build_decision_context(current_book, *selected, {"AAA": {"source_status": "research_only", "analysis_date": "2026-11-17"}})
        row = continuity["comparisons"][0]
        assert row["quantity_change"] == -3 and row["prior_effective_hypothetical_shares"] == 2
        assert "do not prove fills" in row["execution_attribution"]
        assert row["prior_target_shares"] == 8 and row["prior_validation_status"] == "accepted"
        assert row["quantity_change"] == -3 and row["snapshot_chronology"] == "current_newer"
        assert previous_review(root, "2026-11-17", "130000_aaaaaa", "2026-11-17T18:00:00Z")[0] == latest
        invalid_prior = json.loads(malformed_report_path.read_text())
        assert build_decision_context(current_book, malformed, invalid_prior)["baseline_status"] == "unavailable"
        for field, invalid_value in (("changes_vs_draft", {"bad": "shape"}),
                                     ("cash_decision", {"deploy_conditions": "not-a-list"})):
            invalid_copy = json.loads((latest / "portfolio_report.json").read_text())
            invalid_copy["final"][field] = invalid_value
            assert build_decision_context(current_book, latest, invalid_copy)["baseline_status"] == "unavailable"

    def continuity(prior_quantity, prior_time, current_quantity, current_time, *,
                   prior_account="acct", current_account="acct", current_positions_present=True):
        prior_account_row = {"account_id": prior_account, "fetched_at": prior_time,
                             "positions": ([{"ticker": "XLP", "quantity": prior_quantity}]
                                           if prior_quantity is not None else [])}
        prior = {"date": "2026-11-17", "stats": {"accounts": [prior_account_row]},
                 "final": {"trades": [{"ticker": "XLP", "account_id": prior_account,
                                         "action": "BUY", "shares": 12,
                                         "execution_intent": "current"}]},
                 "final_check": {"validated": True, "trades": [{"ticker": "XLP", "account_id": prior_account,
                                                                      "status": "accepted", "effective_shares": 12}]}}
        current_account_row = {"account_id": current_account, "fetched_at": current_time}
        if current_positions_present:
            current_account_row["positions"] = ([{"ticker": "XLP", "quantity": current_quantity}]
                                                 if current_quantity is not None else [])
        current_book = {"accounts": [current_account_row]}
        result = build_decision_context(current_book, Path("2026-11-17/120000_abcdef"), prior)
        return next(row for row in result["comparisons"] if row["ticker"] == "XLP")

    newer = continuity(None, "2026-11-17T10:00:00Z", 12, "2026-11-17T11:00:00Z")
    assert newer["prior_observed_shares"] == 0 and newer["observed_shares"] == 12
    assert newer["quantity_change"] == 12 and newer["snapshot_chronology"] == "current_newer"
    assert "do not prove fills" in newer["execution_attribution"]
    older = continuity(10, "2026-11-17T11:00:00Z", 12, "2026-11-17T10:00:00Z")
    assert older["quantity_change"] is None and older["comparison_status"] == "current_snapshot_older"
    equal = continuity(10, "2026-11-17T11:00:00.123456Z", 12, "2026-11-17T06:00:00.123456-05:00")
    assert equal["snapshot_chronology"] == "same_instant" and equal["quantity_change"] is None
    unknown = continuity(10, "2026-11-17T11:00:00Z", 12, "2026-11-17T12:00:00")
    assert unknown["snapshot_chronology"] == "unknown" and unknown["quantity_change"] is None
    precise_unknown = continuity(10, "2026-11-17T11:00:00.123456789Z", 12, "2026-11-17T12:00:00Z")
    assert precise_unknown["snapshot_chronology"] == "unknown"
    full_exit = continuity(12, "2026-11-17T10:00:00Z", None, "2026-11-17T11:00:00Z")
    assert full_exit["observed_shares"] == 0 and full_exit["quantity_change"] == -12
    missing_account = continuity(12, "2026-11-17T10:00:00Z", None, "2026-11-17T11:00:00Z",
                                 prior_account="old", current_account="new")
    assert missing_account["observed_shares"] is None and missing_account["comparison_status"] == "account_unavailable"
    incomplete = continuity(12, "2026-11-17T10:00:00Z", None, "2026-11-17T11:00:00Z",
                            current_positions_present=False)
    assert incomplete["observed_shares"] is None and incomplete["comparison_status"] == "position_snapshot_incomplete"

    # Run path dates and IDs are New York local time. Compare a prior report's
    # actual generated instant to the current run start in UTC, so a 20:00 NY
    # review after UTC midnight remains eligible, while a truly future report
    # is excluded even if its local run directory sorts earlier.
    with tempfile.TemporaryDirectory(prefix="fm_utc_boundary_") as temp:
        root = Path(temp)
        def completed(run, generated):
            folder = root / "2026-11-17" / run
            folder.mkdir(parents=True)
            report = {"date": "2026-11-17", "run_id": run, "generated_utc": generated,
                      "review_outcome": "complete", "final": {"trades": []},
                      "final_check": {"validated": True, "trades": []},
                      "stats": {"accounts": []}}
            (folder / "portfolio_report.json").write_text(json.dumps(report))
            return folder
        after_midnight = completed("200000_111111", "2026-11-18T01:00:00Z")  # 20:00 EST, Nov 17
        completed("203000_222222", "2026-11-18T03:00:00Z")  # future vs 21:00 EST boundary
        chosen = previous_review(root, "2026-11-17", "210000_333333", "2026-11-18T02:00:00Z")
        assert chosen and chosen[0] == after_midnight

    # The repeated 01:00 hour on the New York fall-back date is ordered by
    # generated instants, not ambiguous wall-clock directory names.
    with tempfile.TemporaryDirectory(prefix="fm_dst_fallback_") as temp:
        root = Path(temp)
        ny = ZoneInfo("America/New_York")
        def dst_report(utc, suffix):
            instant = datetime.fromisoformat(utc.replace("Z", "+00:00"))
            local_run = instant.astimezone(ny).strftime("%H%M%S") + "_" + suffix
            folder = root / "2026-11-01" / local_run
            folder.mkdir(parents=True)
            report = {"date": "2026-11-01", "run_id": local_run, "generated_utc": utc,
                      "review_outcome": "complete", "final": {"trades": []},
                      "final_check": {"validated": True, "trades": []},
                      "stats": {"accounts": []}}
            (folder / "portfolio_report.json").write_text(json.dumps(report))
            return folder, local_run
        first_fold, _ = dst_report("2026-11-01T05:30:00Z", "111111")  # 01:30 EDT
        second_fold_latest, _ = dst_report("2026-11-01T06:15:00Z", "222222")  # 01:15 EST
        dst_report("2026-11-01T06:30:00Z", "333333")  # 01:30 EST, future to current run
        current = datetime.fromisoformat("2026-11-01T06:20:00+00:00")  # 01:20 EST
        current_run = current.astimezone(ny).strftime("%H%M%S") + "_444444"
        first_only_root = root / "only_first_fold"
        first_only = first_only_root / "2026-11-01" / first_fold.name
        first_only.mkdir(parents=True)
        (first_only / "portfolio_report.json").write_bytes((first_fold / "portfolio_report.json").read_bytes())
        first_only_choice = previous_review(first_only_root, "2026-11-01", current_run, current)
        assert first_only_choice and first_only_choice[0] == first_only
        chosen = previous_review(root, "2026-11-01", current_run, current)
        assert chosen and chosen[0] == second_fold_latest and chosen[0] != first_fold
        assert chosen[1]["_continuity_selection_basis"] == "utc_timestamp"

    _check_nov14_to_latest_nov17()

    print("check_decision_contract: passed")


if __name__ == "__main__":
    main()
