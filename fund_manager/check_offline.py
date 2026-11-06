"""Free checks for the fund-manager add-on. No network and no LLM calls."""
from __future__ import annotations

import sys
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
    assert "opus" not in fm.QUICK_MODEL
    assert fm._without_opus("anthropic/claude-opus-5.5", False, "conservative") == fm.QUICK_MODEL
    assert fm._without_opus("anthropic/claude-opus-5.5", True, "conservative") == "anthropic/claude-opus-5.5"

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
    assert "below the $500 minimum" in text, text
    assert "exceeds the 12% cap" in text, text  # NVDA stays ~50%
    assert "AI look-through" in text, text
    assert "ATR fallback" in text, text  # NVDA and AAPL have no plan stop
    assert "earnings" in text, text
    risk = check["post_trade_risk"]
    assert risk["beta_spy"] > 1, risk
    assert "SPY_-10pct_beta_scaled" in risk["stress"]
    assert risk["stress"]["SPY_-10pct_beta_scaled"]["pnl_usd"] < 0
    spy = next(a for a in check["post_trade"] if a["ticker"] == "SPY")
    assert spy["downside_source"] == "plan"
    nvda = next(a for a in check["post_trade"] if a["ticker"] == "NVDA")
    assert nvda["downside_source"] == "atr_fallback"
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
    print("check_offline: ok")
    print(f"  speaking {order_a} vs {order_b}")
    print(f"  post-trade beta {risk['beta_spy']} AI {risk['ai_lookthrough_pct']}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
