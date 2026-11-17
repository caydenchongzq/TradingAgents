#!/usr/bin/env python3
"""Offline checks for the account book contract and consolidated projection."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from portfolio_book import PortfolioBookError, normalize_book, replace_account


legacy = normalize_book({"cash": 400, "positions": [{"ticker": "aapl", "quantity": 2, "average_price": 100}]})
assert legacy["schema_version"] == 2 and len(legacy["accounts"]) == 1
assert legacy["accounts"][0]["account_id"] == "legacy"
assert legacy["accounts"][0]["available_cash"] is None
assert legacy["positions"][0]["ticker"] == "AAPL"

book = normalize_book({"schema_version": 2, "accounts": [
    {"account_id": "sg", "broker": "moomoo", "currency": "USD", "cash": None,
     "available_cash": None, "positions": [{"ticker": "AAPL", "quantity": 5, "average_price": 120,
                                              "cost_method": "average_cost", "cost_source": "moomoo"}]},
    {"account_id": "us", "broker": "tiger", "currency": "USD", "cash": 600,
     "available_cash": 500, "positions": [{"ticker": "AAPL", "quantity": 10, "average_price": 100,
                                              "cost_method": "average_cost", "cost_source": "tiger"}]},
]})
pos = book["positions"][0]
assert pos["quantity"] == 15 and math.isclose(pos["average_price"], 106.6666667, rel_tol=1e-7)
assert pos["account_positions"][0]["account_id"] == "sg"
assert book["cash"] is None  # never total partially known cash

unknown_cost = normalize_book({"schema_version": 2, "accounts": [
    {"account_id": "a", "currency": "USD", "cash": 500, "positions": [
        {"ticker": "AAA", "quantity": 10, "average_price": 100, "cost_method": "average_cost"}]},
    {"account_id": "b", "currency": "USD", "cash": 500, "positions": [
        {"ticker": "AAA", "quantity": 5, "average_price": None, "cost_method": None}]},
]})
assert unknown_cost["positions"][0]["quantity"] == 15
assert unknown_cost["positions"][0]["average_price"] is None

incompatible_cost = normalize_book({"schema_version": 2, "accounts": [
    {"account_id": "a", "currency": "USD", "cash": 100, "positions": [
        {"ticker": "AAA", "quantity": 10, "average_price": 100, "cost_method": "average_cost"}]},
    {"account_id": "b", "currency": "USD", "cash": 100, "positions": [
        {"ticker": "AAA", "quantity": 5, "average_price": 120, "cost_method": "diluted_cost"}]},
]})
assert incompatible_cost["positions"][0]["average_price"] is None

isolated = replace_account(book, {"account_id": "us", "broker": "tiger", "currency": "USD", "cash": 900,
                                  "positions": [{"ticker": "MSFT", "quantity": 3, "average_price": 200}]})
assert isolated["accounts"][0]["positions"][0]["ticker"] == "AAPL"  # SG account preserved
assert isolated["accounts"][1]["positions"][0]["ticker"] == "MSFT"
repeated = replace_account(isolated, {"account_id": "us", "broker": "tiger", "currency": "USD", "cash": 900,
                                     "positions": [{"ticker": "MSFT", "quantity": 3, "average_price": 200}]})
assert len(repeated["accounts"]) == 2 and len(repeated["positions"]) == 2

try:
    normalize_book({"schema_version": 2, "accounts": [{"account_id": "x", "cash": 1, "positions": [
        {"ticker": "AAA", "quantity": 1, "average_price": 1},
        {"ticker": "AAA", "quantity": 2, "average_price": 1},
    ]}]})
except PortfolioBookError:
    pass
else:
    raise AssertionError("duplicate same-account instrument accepted")

for bad_scope in (
    {"account_id": "sgd", "currency": "SGD", "cash": 100, "positions": []},
    {"account_id": "sg-stock", "currency": "USD", "cash": 100,
     "positions": [{"ticker": "D05", "market": "SG", "currency": "SGD", "quantity": 1}]},
):
    try:
        normalize_book({"schema_version": 2, "accounts": [bad_scope]})
    except PortfolioBookError:
        pass
    else:
        raise AssertionError("out-of-scope cash or position was included in USD projection")

for bad_version in (
    {"schema_version": 3, "positions": [{"ticker": "AAA", "quantity": 1}]},
    {"accounts": [{"account_id": "acct", "currency": "USD", "cash": 5, "positions": []}]},
):
    try:
        normalize_book(bad_version)
    except PortfolioBookError:
        pass
    else:
        raise AssertionError("unsupported or unversioned canonical account book was migrated as legacy")

print("portfolio_book checks passed")
