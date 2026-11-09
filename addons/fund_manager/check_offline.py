"""Free checks for the fund-manager add-on. No network and no LLM calls."""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import market_calendar as cal
import fund_manager as fm
import preflight


def _stats():
    return {
        "account": {"cash": 10000.0, "total_value": 100000.0},
        "cash_policy": {"min_pct": 5, "max_pct": 35},
        "positions": [
            {"ticker": "NVDA", "quantity": 100, "price": 500, "theme": "ai_growth",
             "ann_vol_pct": 45, "obs_1y": 252, "stress_beta": 1.8, "atr_stop": 450,
             "type": "EQUITY", "earnings": {"date": "2026-11-12", "calendar_days": 3}},
            {"ticker": "SPY", "quantity": 40, "price": 500, "theme": "core_index",
             "ann_vol_pct": 14, "obs_1y": 252, "stress_beta": 1.0, "atr_stop": 470,
             "type": "ETF"},
            {"ticker": "AAPL", "quantity": 20, "price": 200, "theme": "megacap_quality",
             "ann_vol_pct": 22, "obs_1y": 252, "stress_beta": 1.2, "atr_stop": 180,
             "type": "EQUITY"},
        ],
        "watchlist": [],
        "risk_policy": dict(fm.RISK_POLICY),
    }


def main() -> int:
    normalized_system = " ".join(fm.FM_SYSTEM.split())
    assert "not inherited percentage targets or owner-imposed limits" in normalized_system
    assert "without trying to satisfy an arbitrary percentage" in normalized_system
    assert "single_stock_cap_pct" not in fm.RISK_POLICY and "top3_cap_pct" not in fm.RISK_POLICY
    assert "min_trade_usd" not in fm.RISK_POLICY
    assert "do not short, use options or margin, spend more cash than is" in fm.FM_SYSTEM
    assert "Stay inside them" not in fm.FM_SYSTEM
    assert "an upcoming earnings date is a risk factor, not an automatic ban" in normalized_system
    assert "assess gap/event risk" in normalized_system
    assert fm.source_freshness({"analysis_date": "2026-11-06"}, "2026-11-06")["status"] == "current"
    assert fm.source_freshness({"analysis_date": "2026-11-02"}, "2026-11-06")["status"] == "reused"
    assert fm.source_freshness({"analysis_date": "2026-10-28"}, "2026-11-06")["status"] == "stale"
    assert fm.source_freshness(None, "2026-11-06")["status"] == "missing"
    with tempfile.TemporaryDirectory(prefix="fm_scout_freshness_") as temp:
        scout_dir = Path(temp)
        for day, headline in (("2026-11-02", "older"), ("2026-11-08", "latest"),
                              ("2026-11-12", "future")):
            folder = scout_dir / day
            folder.mkdir()
            ranking = [{"ticker": f"S{i:03d}", "rank": i + 1, "bucket": "holding" if i < 12 else "candidate"}
                       for i in range(166)] if day == "2026-11-08" else [{"ticker": "AAA", "rank": 1}]
            (folder / "report.json").write_text(json.dumps({
                "date": day, "generated_utc": day + "T12:00:00Z", "headline": headline,
                "ranking": ranking, "ranking_status": "partial" if day == "2026-11-08" else "complete",
                "ranking_count": len(ranking), "ranking_expected_count": 166, "changes": [],
            }), encoding="utf-8")
        scout = fm.load_scout_report(scout_dir, "2026-11-10")
        assert scout["status"] == "reused" and scout["date"] == "2026-11-08" and scout["age_days"] == 2, scout
        assert scout["report"]["headline"] == "latest" and scout["report"]["ranking"][0]["ticker"] == "S000"
        assert len(scout["report"]["ranking"]) == 166
        assert scout["report"]["ranking_status"] == "partial"
        assert fm.load_scout_report(scout_dir, "2026-11-20")["status"] == "stale"
        assert fm.load_scout_report(scout_dir / "absent", "2026-11-10")["status"] == "missing"
        watch_path = scout_dir / "watchlist.json"
        for key in ("items", "candidates", "watchlist", "tickers"):
            watch_path.write_text(json.dumps({key: [{"ticker": "ZZZ"}]}), encoding="utf-8")
            assert fm.load_watchlist(watch_path, [])[0]["ticker"] == "ZZZ"
        watch_path.write_text(json.dumps({"items": [], "candidates": [{"ticker": "SHOULD_NOT_FALLBACK"}]}), encoding="utf-8")
        assert fm.load_watchlist(watch_path, []) == []

    assert fm.approved_target_weights({"weights_pct": {"AAA": 40}}) == {}
    assert fm.approved_target_weights({"approved_targets_pct": {"AAA": 40, "bad": 120, "bool": True}}) == {"AAA": 40.0}
    guide = {"portfolio_guidance": {"cash_reference_pct": 10, "long_term_pct": 30, "swing_pct": 60}}
    assert fm.approved_portfolio_guidance(guide) == {
        "cash_reference_pct": 10.0, "long_term_pct": 30.0, "swing_pct": 60.0,
        "cash_is_soft": True, "force_rebalance": False,
        "sector_weights_are_guidelines": True, "swing_horizon": "flexible",
    }
    assert fm.approved_portfolio_guidance({"portfolio_guidance": {
        "cash_reference_pct": 15, "long_term_pct": 30, "swing_pct": 60}}) is None
    alt_guide = {"portfolio_guidance": {"cash_reference_pct": 5, "long_term_pct": 45, "swing_pct": 50}}
    alt_context = fm.build_context(
        {"cash": 5000, "positions": []}, alt_guide,
        {"positions": [], "watchlist": [], "cash_policy": {"min_pct": 0, "max_pct": 100}},
        {}, None, "2026-11-06")
    assert "5% cash, 45% long-term and 50% swing" in alt_context
    assert "Cash can be below 5%, including zero" in alt_context
    policy_context = fm.build_context(
        {"cash": 10000, "positions": []}, guide,
        {"positions": [], "watchlist": [], "cash_policy": {"min_pct": 0, "max_pct": 100}},
        {}, None, "2026-11-06")
    assert "10% cash, 30% long-term and 60% swing" in policy_context
    assert "Cash can be below 10%, including zero" in policy_context
    assert "No automatic trades or forced rebalancing" in policy_context
    assert fm.trading_sleeve("BRK-B", {"type": "EQUITY"}) == "swing"
    assert fm.trading_sleeve("UNKNOWN", {"type": None}) == "unclassified"
    assert cal.is_trading_day(cal.date(2026, 11, 6))
    assert not cal.is_trading_day(cal.date(2026, 11, 7))  # Saturday
    assert not cal.is_trading_day(cal.date(2026, 11, 26))  # Thanksgiving
    assert cal.previous_trading_day(cal.date(2026, 11, 7)) == cal.date(2026, 11, 6)

    noon = cal.at_ny(cal.date(2026, 11, 6), datetime.min.time().replace(hour=12))
    info = cal.session_info(noon)
    assert info["market_state"] == "regular", info
    assert info["last_completed_session"] == "2026-11-05"
    assert info["ny_date"] == "2026-11-06"
    assert cal.expected_bar_for(cal.date(2026, 11, 6), noon) == cal.date(2026, 11, 5)

    after = cal.at_ny(cal.date(2026, 11, 6), datetime.min.time().replace(hour=17))
    assert cal.session_info(after)["market_state"] == "after_close"
    assert cal.expected_bar_for(cal.date(2026, 11, 6), after) == cal.date(2026, 11, 6)

    wall = cal.at_ny(cal.date(2026, 11, 6), datetime.min.time().replace(hour=16))
    aware = wall if wall.tzinfo else wall.astimezone()
    assert aware.utcoffset().total_seconds() == -5 * 3600, aware

    short = fm._trend_stats(fm.pd.Series(range(1, 102), dtype=float))
    assert short["history_bars"] == 101
    assert "sma200" not in short
    assert "ret_1y_pct" not in short
    assert short["ret_1d_pct"] is not None and short["ret_5d_pct"] is not None
    long = fm._trend_stats(fm.pd.Series(range(1, 260), dtype=float))
    assert "sma200" in long and "ret_1y_pct" in long

    order_a = fm.speaking_order(1, "2026-11-06")
    order_b = fm.speaking_order(1, "2026-11-07")
    assert order_a != order_b, (order_a, order_b)
    assert set(order_a) == {"aggressive", "conservative", "neutral"}
    assert fm.speaking_order(2, "2026-11-06")[0] != order_a[0]

    assert fm.QUICK_MODEL == "openai/gpt-6-luna"
    assert fm.CONSERVATIVE_MODEL == "anthropic/claude-opus-5.5"
    assert fm._without_opus("anthropic/claude-opus-5.5", False, "conservative") == "anthropic/claude-opus-5.5"
    assert fm.opus_on_other_seat("anthropic/claude-opus-5.5", "neutral")
    assert not fm.opus_on_other_seat("anthropic/claude-opus-5.5", "conservative")
    assert fm._without_opus("anthropic/claude-opus-5.5", False, "neutral") == "anthropic/claude-opus-5.5"
    assert fm.sleeve_fraction("AAPL", "ai_growth", fm.DEFAULT_AI_SLEEVE) == 0.0
    assert fm.sleeve_fraction("MSFT", "megacap_quality", fm.DEFAULT_AI_SLEEVE) == 1.0
    assert abs(fm.sleeve_fraction("SPY", "core_index", fm.DEFAULT_AI_SLEEVE) - 0.33) < 1e-9
    half = fm.resolve_ai_sleeve({"ai_sleeve": {"ticker_weight": {"AAPL": 0.5}}})
    assert half["ticker_weight"]["AAPL"] == 0.5 and half["ticker_weight"]["SPY"] == 0.33
    assert "ai_sleeve_cap_pct" not in fm._policy_with_sleeve({})
    assert "cap_pct" not in fm.resolve_ai_sleeve({"ai_sleeve": {"cap_pct": 42}})
    table = fm.subtheme_table([
        {"ticker": "NVDA", "sub_theme": "ai_compute_semis", "weight_pct": 12.5, "market_value": 1000},
        {"ticker": "AMD", "sub_theme": "ai_compute_semis", "weight_pct": 2.5, "market_value": 200},
        {"ticker": "AAPL", "sub_theme": None, "weight_pct": 3, "market_value": 300},
    ], {"subthemes": {"ai_compute_semis": ["NVDA", "AMD"], "ai_energy": ["VST"]},
        "subtheme_labels": {"ai_compute_semis": "AI compute & semis", "ai_energy": "AI energy"}})
    by = {row["id"]: row for row in table}
    assert by["ai_compute_semis"]["weight_pct"] == 15.0
    assert by["ai_energy"]["weight_pct"] == 0.0
    assert fm.subtheme_of("NVDA", {"subthemes": {"ai_compute_semis": ["NVDA"]}}) == "ai_compute_semis"

    decision = {
        "cash_decision": {"target_cash_pct": 20},
        "levels": [{"ticker": "SPY", "stop_or_reassess": 470}],
        "trades": [
            {"ticker": "NVDA", "action": "SELL", "shares": 1, "decision": "trim"},
            {"ticker": "AAPL", "action": "HOLD", "shares": 0, "decision": "keep"},
            {"ticker": "SPY", "action": "HOLD", "shares": 0, "decision": "keep"},
        ],
    }
    decision["trades"][1] = {"ticker": "AAPL", "action": "BUY", "shares": 1, "decision": "add"}
    check = fm.validate_trades(decision, _stats())
    text = " ".join(check["warnings"])
    assert "trade-size indicator" not in text, text
    assert check["ending_cash"] == 10300.0, check["ending_cash"]
    assert "advisory position-size indicator" not in text, text
    assert "AI look-through" not in text and "top 3" not in text, text
    assert "ATR fallback" in text, text  # NVDA and AAPL have no plan stop
    assert "earnings" in text, text
    risk = check["post_trade_risk"]
    assert risk["beta_spy"] > 1, risk
    assert abs(risk["ai_lookthrough_pct"] - 56.1) < 0.15, risk
    assert risk["top3_pct"] > 50, risk
    assert next(a for a in check["post_trade"] if a["ticker"] == "NVDA")["weight_after_pct"] > 40
    assert not any(key.endswith("cap_pct") for key in check["post_trade_risk"])
    half_stats = _stats()
    half_stats["risk_policy"] = {**fm.RISK_POLICY, "ai_sleeve": half}
    half_ai = fm.validate_trades(decision, half_stats)["post_trade_risk"]["ai_lookthrough_pct"]
    assert abs(half_ai - 58.2) < 0.15, half_ai
    assert "SPY_-10pct_beta_scaled" in risk["stress"]
    assert risk["stress"]["SPY_-10pct_beta_scaled"]["pnl_usd"] < 0

    # A small affordable trade is valid; available cash remains a real constraint.
    small_trade = fm.validate_trades({
        "cash_decision": {"target_cash_pct": 10},
        "levels": [],
        "trades": [{"ticker": "AAPL", "action": "BUY", "shares": 1, "decision": "add"}],
    }, _stats())
    assert small_trade["ending_cash"] == 9800.0, small_trade["ending_cash"]
    assert "trade-size indicator" not in " ".join(small_trade["warnings"])
    overspend = fm.validate_trades({
        "cash_decision": {"target_cash_pct": 10},
        "levels": [],
        "trades": [{"ticker": "AAPL", "action": "BUY", "shares": 51, "decision": "add"}],
    }, _stats())
    assert any("overspend cash by $200" in warning for warning in overspend["warnings"]), overspend["warnings"]
    spy = next(a for a in check["post_trade"] if a["ticker"] == "SPY")
    assert spy["downside_source"] == "plan"
    nvda = next(a for a in check["post_trade"] if a["ticker"] == "NVDA")
    assert nvda["downside_source"] == "atr_fallback"

    # History remains available as evidence, but no inherited short-history size cap is applied.
    hist = _stats()
    hist["positions"] = [
        {"ticker": "MSFT", "quantity": 10, "price": 500, "theme": "megacap_quality",
         "ann_vol_pct": 22, "obs_1y": 248, "history_bars": 501, "stress_beta": 1.1,
         "atr_stop": 450, "type": "EQUITY"},
        {"ticker": "SPCX", "quantity": 30, "price": 100, "theme": "ai_growth",
         "ann_vol_pct": 40, "obs_1y": 101, "history_bars": 103, "stress_beta": 1.5,
         "atr_stop": 80, "type": "EQUITY"},
    ]
    hist_decision = {
        "cash_decision": {"target_cash_pct": 20},
        "levels": [
            {"ticker": "MSFT", "stop_or_reassess": 450},
            {"ticker": "SPCX", "stop_or_reassess": 80},
        ],
        "trades": [
            {"ticker": "MSFT", "action": "HOLD", "shares": 0, "decision": "keep"},
            {"ticker": "SPCX", "action": "HOLD", "shares": 0, "decision": "keep"},
        ],
    }
    hist_check = fm.validate_trades(hist_decision, hist)
    hist_text = " ".join(hist_check["warnings"])
    assert "advisory position-size indicator" not in hist_text, hist_text
    assert "history 103 bars" not in hist_text, hist_text
    assert any(row["ticker"] == "SPCX" and row["weight_after_pct"] == 3.0
               for row in hist_check["post_trade"]), hist_check["post_trade"]
    bad = fm.pd.DataFrame({
        "Date": ["2026-11-04 00:00:00-05:00", "2026-11-05 00:00:00-04:00"],
        "Close": [200.0, None],
    })
    close, last = preflight._bar_close(bad, cal.date(2026, 11, 5))
    assert close is None and last == cal.date(2026, 11, 5), (close, last)
    good = bad.copy()
    good.loc[1, "Close"] = 181.6
    close, last = preflight._bar_close(good, cal.date(2026, 11, 5))
    assert abs(close - 181.6) < 0.01, close

    sleeve = fm.DEFAULT_AI_SLEEVE
    assert fm.sleeve_fraction("QQQ", "core_index", sleeve) == 0.50
    assert fm.sleeve_fraction("SMH", "ai_growth", sleeve) == 1.0
    assert fm.sleeve_fraction("IWM", "core_index", sleeve) == 0.0
    assert fm.sleeve_fraction("XLE", "sector_etf", sleeve) == 0.0
    assert fm.sleeve_fraction("XLF", "sector_etf", sleeve) == 0.0
    assert fm.sleeve_fraction("GLD", "sector_etf", sleeve) == 0.0
    assert fm.sleeve_fraction("SPY", "core_index", sleeve) == 0.33
    assert fm.is_etf_instrument("DRAM", {"type": "etf"})
    assert fm.is_etf_instrument("QQQ", {})
    assert not fm.is_etf_instrument("NVDA", {"type": "EQUITY"})
    assert not fm.is_etf_instrument("NOTE", {"type": "ETN"})
    assert fm.trading_sleeve("NOTE", {"type": "ETN"}) == "unclassified"
    assert fm.trading_sleeve("SPY", {}) == "core_etf"
    assert fm.trading_sleeve("AAPL", {"type": "EQUITY"}) == "swing"

    def _sleeve_book(etf_value: float):
        total = 100000.0
        stock = 40000.0
        cash = total - etf_value - stock
        return {
            "account": {"cash": cash, "total_value": total},
            "cash_policy": {"min_pct": 5, "max_pct": 35},
            "positions": [
                {"ticker": "SPY", "quantity": etf_value / 100, "price": 100, "type": "ETF",
                 "theme": "core_index", "ann_vol_pct": 14, "history_bars": 501,
                 "stress_beta": 1.0, "atr_stop": 90},
                {"ticker": "AAPL", "quantity": stock / 100, "price": 100, "type": "EQUITY",
                 "theme": "megacap_quality", "ann_vol_pct": 22, "history_bars": 501,
                 "stress_beta": 1.1, "atr_stop": 90},
            ],
            "watchlist": [
                {"ticker": "XLV", "type": "etf", "price": 100, "ann_vol_pct": 16,
                 "history_bars": 501, "stress_beta": 0.7},
            ],
            "risk_policy": dict(fm.RISK_POLICY),
        }

    def _hold(book):
        return {
            "cash_decision": {"target_cash_pct": book["account"]["cash"] / book["account"]["total_value"] * 100},
            "levels": [
                {"ticker": "SPY", "stop_or_reassess": 90},
                {"ticker": "AAPL", "stop_or_reassess": 90},
            ],
            "trades": [
                {"ticker": "SPY", "action": "HOLD", "shares": 0, "decision": "keep"},
                {"ticker": "AAPL", "action": "HOLD", "shares": 0, "decision": "keep"},
            ],
        }

    ok_book = _sleeve_book(26400)
    ok_check = fm.validate_trades(_hold(ok_book), ok_book)
    assert ok_check["sleeves"]["status"] == "ok", ok_check["sleeves"]
    assert abs(ok_check["sleeves"]["core_etf"]["weight_pct"] - 26.4) < 0.05, ok_check["sleeves"]
    assert not any("core ETF" in w for w in ok_check["warnings"]), ok_check["warnings"]
    assert next(a for a in ok_check["post_trade"] if a["ticker"] == "SPY")["sleeve"] == "core_etf"
    assert next(a for a in ok_check["post_trade"] if a["ticker"] == "AAPL")["sleeve"] == "swing"
    assert next(a for a in ok_check["post_trade"] if a["ticker"] == "cash")["sleeve"] == "swing"
    warn_check = fm.validate_trades(_hold(_sleeve_book(31000)), _sleeve_book(31000))
    assert warn_check["sleeves"]["status"] == "ok", warn_check["sleeves"]
    assert warn_check["breaches"] == [], warn_check["breaches"]
    assert not any("core ETF" in w for w in warn_check["warnings"]), warn_check["warnings"]
    breach_check = fm.validate_trades(_hold(_sleeve_book(33000)), _sleeve_book(33000))
    assert breach_check["sleeves"]["status"] == "ok", breach_check["sleeves"]
    assert not any("core ETF" in w for w in breach_check["warnings"]), breach_check["warnings"]
    flexible_book = _sleeve_book(60000)
    flexible_book["account"]["cash"] = 0
    flexible_book["cash_policy"] = {"min_pct": 0, "max_pct": 100, "soft_default_pct": 10,
                                     "portfolio_guidance": fm.approved_portfolio_guidance(guide)}
    flexible_plan = _hold(flexible_book)
    flexible_plan["cash_decision"]["target_cash_pct"] = 0
    flexible_check = fm.validate_trades(flexible_plan, flexible_book)
    assert flexible_check["ending_cash_pct"] == 0 and not flexible_check["breaches"], flexible_check
    assert not any("cash" in warning.lower() and "outside" in warning.lower()
                   for warning in flexible_check["warnings"]), flexible_check["warnings"]
    etf_only = _sleeve_book(20000)
    no_stop = {
        "cash_decision": {"target_cash_pct": 20},
        "levels": [{"ticker": "AAPL", "stop_or_reassess": 90}],
        "trades": [
            {"ticker": "XLV", "action": "BUY", "shares": 10, "decision": "new_position"},
            {"ticker": "AAPL", "action": "HOLD", "shares": 0, "decision": "keep"},
        ],
    }
    # Drop SPY so the buy is the only ETF and still under the sleeve cap.
    etf_only["positions"] = [p for p in etf_only["positions"] if p["ticker"] != "SPY"]
    etf_only["account"]["cash"] = 60000
    loose = fm.validate_trades(no_stop, etf_only)
    loose_text = " ".join(loose["warnings"])
    assert "XLV: no stop" not in loose_text, loose_text
    assert next(t for t in loose["trades"] if t["ticker"] == "XLV")["sleeve"] == "core_etf"

    context = fm.build_context(
        {"cash": 0, "positions": []},
        {"weights_pct": {"EXAMPLE": 90}, "approved_targets_pct": {"AAA": 25},
         "owner_notes": "legacy unreviewed instruction", "custom_policy": {"target": 77}},
        {"positions": [], "watchlist": []}, {}, None, "2026-11-06")
    assert "EXAMPLE" not in context and '"weights_pct"' not in context, context
    assert "legacy unreviewed instruction" not in context and '"custom_policy"' not in context, context
    assert '"approved_targets_pct"' in context and '"AAA": 25' in context, context
    assert "held for weeks to months" not in fm.FM_SYSTEM and "held for weeks to months" not in fm.ANALYST_TASK
    assert "otherwise do not invent allocation percentages" in fm.FM_SYSTEM
    assert "Do not apply swing-trading rules or tight stops to ETFs" not in fm.FM_SYSTEM
    assert "Do not apply swing-trading rules or tight stops to ETFs" not in fm.ANALYST_TASK
    assert "investment_role" in fm.TRADE_ITEM["properties"]
    scout_context = fm.build_context(
        {"cash": 0, "positions": []}, {},
        {"positions": [], "watchlist": [], "report_sources": {"AAA": {"status": "stale", "source_date": "2026-10-28"}}},
        {}, None, "2026-11-06", {"status": "reused", "date": "2026-11-02", "age_days": 4,
                                 "report": {"ranking": [{"ticker": "AAA", "rank": 1}]}})
    assert '"status": "stale"' in scout_context and '"status": "reused"' in scout_context
    assert '"ticker": "AAA"' in scout_context and "Dated theme scout report" in scout_context
    rendered = []
    fm.render_plan_trades(rendered, {"priority_order": []}, {
        "trades": [], "post_trade": [{"ticker": "AAA", "shares_after": 1, "value_after": 100,
                                        "weight_after_pct": 25, "is_new": False}], "warnings": [],
        "ending_cash": 0, "ending_cash_pct": 0,
    }, {"weights_pct": {"EXAMPLE": 90}, "approved_targets_pct": {"AAA": 25}})
    rendered_text = "\n".join(rendered)
    assert "Owner-approved target" in rendered_text and "25.0%" in rendered_text
    assert "EXAMPLE" not in rendered_text and "Target (guidance)" not in rendered_text
    import pandas as pd
    idx = pd.bdate_range("2025-10-01", "2025-11-06")
    close = pd.DataFrame({"SPY": [100 + i for i in range(len(idx))]}, index=idx)
    quotes = fm.live_quotes(["SPY"], close, "2025-11-06")
    assert quotes["SPY"]["timing"] == "as_of", quotes
    assert quotes["SPY"]["price"] == close["SPY"].iloc[-1]
    assert "live" not in (quotes["SPY"]["source"] or "")
    print("check_offline: ok")
    print(f"  configured safeguard states {warn_check['sleeves']['status']}/{breach_check['sleeves']['status']}")
    print(f"  speaking {order_a} vs {order_b}")
    print(f"  post-trade beta {risk['beta_spy']} AI {risk['ai_lookthrough_pct']}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
