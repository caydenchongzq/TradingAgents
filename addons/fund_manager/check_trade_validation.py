"""Offline policy/account scenarios; no market, model, SDK or user-data access."""
from __future__ import annotations
from copy import deepcopy
import math
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trade_validation import TradePolicyError, parse_trade_policy, validate_trades
from portfolio_book import normalize_book


def fixture(cash=10000, quantity=15, price=100, sellable=15, available=10000):
    return {"accounts": [{"account_id": "a", "broker": "fixture", "cash": cash,
                          "available_cash": available, "positions": [
                              {"ticker": "AAA", "quantity": quantity, "price": price,
                               "average_price": 80, "sellable_quantity": sellable}]}],
            "positions": [{"ticker": "AAA", "price": price}],
            "watchlist": [{"ticker": "NEW", "price": 100}, {"ticker": "OTHER", "price": 200}]}


def order(action, shares, ticker="AAA", **extra):
    return {"account_id": "a", "ticker": ticker, "action": action, "shares": shares, **extra}


def run(rows, stats=None):
    stats = fixture() if stats is None else stats
    before = deepcopy(stats)
    decision = {"trades": deepcopy(rows)}
    result = validate_trades(decision, stats)
    assert stats == before and decision["trades"] == rows
    return result


def rejected(result):
    json.dumps(result, allow_nan=False)
    assert all(row["status"] == "rejected" for row in result["trades"]), result
    assert result["fees_usd"] == 0
    assert all(row["shares"] == 0 and row["cash_delta_usd"] == 0 for row in result["trades"])


def main():
    assert parse_trade_policy({}) == {"fee_per_order_usd": 3, "min_buy_notional_usd": 1000,
                                      "min_residual_notional_usd": 1000}
    for policy in (None, [], "bad", {"fee_per_order_usd": -1}, {"fee_per_order_usd": True},
                   {"fee_per_order_usd": math.inf}, {"min_buy_notional_usd": 0},
                   {"min_residual_notional_usd": -1}, {"min_buy_notional_usd": math.nan}):
        try:
            parse_trade_policy({"trade_policy": policy})
        except TradePolicyError:
            pass
        else:
            raise AssertionError("Invalid present policy accepted")
    assert parse_trade_policy({"trade_policy": {"fee_per_order_usd": 0}})["fee_per_order_usd"] == 0
    # Same floor applies to both roles and to both new positions/top-ups.
    for role in ("swing", "long_term"):
        for ticker, intent in (("AAA", "add"), ("NEW", "new_position")):
            rejected(run([order("BUY", 9, ticker, investment_role=role, decision=intent)]))
            exact = run([order("BUY", 10, ticker, investment_role=role, decision=intent)],
                        fixture(cash=1003, available=1003))
            assert exact["trades"][0]["status"] == "accepted" and exact["ending_cash"] == 0
            assert exact["fees_usd"] == 3
            assert any(p["ticker"] == ticker for p in exact["accounts_after"][0]["positions"])
    near = fixture(price=99.9)
    rejected(run([order("BUY", 10)], near))  # $999 precisely, not floating tolerance
    rejected(run([order("BUY", 10)], fixture(cash=1002, available=1002)))
    rejected(run([order("BUY", 10)], fixture(cash=10000, available=1002)))
    # Residual rule is strict < $1000, with a visible effective full exit.
    adjusted = run([order("SELL", 10, decision="trim")])
    assert adjusted["trades"][0]["status"] == "adjusted"
    assert adjusted["trades"][0]["shares"] == 15 and adjusted["trades"][0]["fully_closed"]
    exact_residual = run([order("SELL", 5, decision="trim")])
    assert exact_residual["trades"][0]["shares"] == 5 and exact_residual["trades"][0]["status"] == "accepted"
    explicit = run([order("SELL", 5, decision="exit")])
    assert explicit["trades"][0]["shares"] == 15 and explicit["trades"][0]["status"] == "adjusted"
    small = run([order("SELL", 6, decision="exit")], fixture(quantity=6, sellable=6, cash=0, available=0))
    assert small["trades"][0]["status"] == "accepted" and small["ending_cash"] == 597
    fractional = run([order("SELL", 15.5, decision="exit")], fixture(quantity=15.5, sellable=15.5))
    assert fractional["trades"][0]["shares"] == 15.5 and fractional["trades"][0]["fully_closed"]
    limited = run([order("SELL", 1, decision="exit")], fixture(quantity=15.5, sellable=10.5))
    assert limited["trades"][0]["shares"] == 10.5 and not limited["trades"][0]["fully_closed"]
    assert "locked residual" in limited["trades"][0]["reason"]
    rejected(run([order("SELL", 1, decision="exit")], fixture(sellable=0)))
    # Order processing permits a same-account sale to fund a different symbol.
    sequence = run([order("BUY", 10, "NEW"), order("SELL", 15, decision="exit")],
                   fixture(cash=0, available=0))
    assert [row["action"] for row in sequence["trades"]] == ["SELL", "BUY"]
    assert [row["status"] for row in sequence["trades"]] == ["accepted", "accepted"]
    assert sequence["ending_cash"] == 494 and sequence["fees_usd"] == 6
    short = run([order("BUY", 10, "NEW"), order("SELL", 10, decision="exit")],
                fixture(cash=0, available=0, quantity=10, sellable=10))
    assert short["trades"][0]["shares"] == 10 and short["trades"][1]["status"] == "rejected"
    assert short["ending_cash"] == 997
    # A second account is not a source of cash, shares or implicit routing.
    multi = fixture(cash=0, available=0)
    multi["accounts"].append({"account_id": "b", "cash": 20000, "positions": []})
    rejected(run([order("BUY", 10, "NEW")], multi))
    rejected(run([{**order("SELL", 15), "account_id": "b"}], multi))
    rejected(run([{k: v for k, v in order("BUY", 10, "NEW").items() if k != "account_id"}], multi))
    unknown = fixture(cash=None, available=None)
    rejected(run([order("SELL", 15, decision="exit"), order("BUY", 10, "NEW")], unknown))
    assert run([order("HOLD", 0)], unknown)["trades"][0]["status"] == "hold"
    # Future conditions remain advice: no quote/cash requirement and no
    # effective shares, fees, cash movement, or sale proceeds.
    deferred_buy = order("BUY", 10, "NEW", decision="new_position", execution_intent="deferred",
                         condition="Only if price closes below 90", target_shares=20)
    deferred_buy_result = run([deferred_buy], fixture(cash=None, available=None))
    buy_row = deferred_buy_result["trades"][0]
    assert buy_row["status"] == "deferred" and buy_row["proposed_shares"] == 10
    assert buy_row["effective_shares"] == buy_row["shares"] == buy_row["fee_usd"] == buy_row["cash_delta_usd"] == 0
    assert deferred_buy_result["ending_cash"] is None and deferred_buy_result["fees_usd"] == 0
    deferred_sell = order("SELL", 5, decision="trim", execution_intent="deferred",
                          condition="After the earnings event", target_shares=10)
    deferred_sell_result = run([deferred_sell], fixture(cash=None, available=None))
    assert deferred_sell_result["trades"][0]["status"] == "deferred"
    assert deferred_sell_result["trades"][0]["cash_delta_usd"] == 0
    assert deferred_sell_result["accounts_after"][0]["positions"][0]["quantity"] == 15
    rejected(run([order("SELL", 16, decision="trim", execution_intent="deferred",
                        condition="Later")]))
    rejected(run([order("BUY", 1, "UNRESEARCHED", decision="new_position",
                        execution_intent="deferred", condition="Later")]))
    rejected(run([order("BUY", 10, "NEW", decision="new_position",
                        execution_intent="deferred", condition=" ")]))
    # Failed/rejected proposals have no accounting effects or model-price escape.
    for value in (-1, 0, True, math.inf, math.nan, .5, "10"):
        rejected(run([order("BUY", value)]))
    rejected(run([order("SELL", 1.5)]))
    rejected(run([order("BUY", 10, "UNRESEARCHED", est_price=100)]))
    rejected(run([order("BUY", 10, order_type="staged")]))
    rejected(run([order("BUY", 10, decision="exit")]))
    rejected(run([order("HOLD", 2)]))
    missing = fixture(price=None)
    rejected(run([order("BUY", 10, est_price=100)], missing))
    duplicates = run([order("BUY", 10), order("SELL", 5)])
    assert next(row for row in duplicates["trades"] if row["action"] == "SELL")["status"] == "rejected"
    assert duplicates["fees_usd"] == 3
    # Two accepted orders reduce unchanged-price marked equity only by fees.
    stats = fixture(cash=8500)
    both = run([order("SELL", 5), order("BUY", 10, "NEW")], stats)
    marked = both["ending_cash"] + sum(p["quantity"] * 100 for p in both["accounts_after"][0]["positions"])
    assert marked == 9994
    unknown_availability = run([order("BUY", 10, "NEW")], fixture(available=None))
    assert "not verified" in unknown_availability["trades"][0]["availability_caveat"]
    legacy = normalize_book({"cash": 10000, "positions": [
        {"ticker": "AAA", "quantity": 1, "average_price": 80}]})
    assert legacy["accounts"][0]["available_cash"] is None
    legacy_check = validate_trades({"trades": [{"account_id": "legacy", "ticker": "AAA",
                                                  "action": "BUY", "shares": 10}]}, {
        "accounts": [{**legacy["accounts"][0], "positions": [{**legacy["accounts"][0]["positions"][0],
                                                                    "price": 100}]}],
        "positions": [{"ticker": "AAA", "price": 100}], "watchlist": [],
    })
    assert legacy_check["trades"][0]["status"] == "accepted"
    assert "not verified" in legacy_check["trades"][0]["availability_caveat"]
    print("trade validation offline check: passed (fees, entries, exits, fractions, accounts, ordering, rejection effects)")


if __name__ == "__main__":
    main()
