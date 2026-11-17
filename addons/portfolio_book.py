"""Account-aware stock book normalization for Argus and TradingAgents add-ons.

The v2 ``accounts`` list is canonical. Top-level positions/cash are a derived
compatibility view for the native TradingAgents CLI and older add-on readers.
This module is deliberately standard-library-only so Argus can import it.
"""
from __future__ import annotations

import math
from typing import Any


class PortfolioBookError(ValueError):
    """The account book cannot be represented safely."""


def _number(value: Any, field: str, *, nullable: bool = False, positive: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool):
        raise PortfolioBookError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PortfolioBookError(f"{field} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise PortfolioBookError(f"{field} must be finite" + (" and positive" if positive else ""))
    return result


def _position(raw: dict[str, Any], account: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PortfolioBookError("position must be an object")
    ticker = str(raw.get("ticker") or "").strip().upper()
    if not ticker:
        raise PortfolioBookError("position ticker is required")
    market = str(raw.get("market") or account.get("market") or "US").strip().upper()
    currency = str(raw.get("currency") or account.get("currency") or "USD").strip().upper()
    quantity = _number(raw.get("quantity"), f"{ticker} quantity")
    assert quantity is not None
    if quantity <= 0:
        raise PortfolioBookError(f"{ticker} quantity must be positive")
    average = _number(raw.get("average_price"), f"{ticker} average_price", nullable=True, positive=True)
    sellable = _number(raw.get("sellable_quantity"), f"{ticker} sellable_quantity", nullable=True)
    if sellable is not None and not 0 <= sellable <= quantity:
        raise PortfolioBookError(f"{ticker} sellable_quantity must be between zero and quantity")
    diluted = _number(raw.get("diluted_cost"), f"{ticker} diluted_cost", nullable=True)
    return {
        **{k: v for k, v in raw.items() if k not in {
            "ticker", "market", "currency", "quantity", "average_price", "sellable_quantity",
            "diluted_cost", "cost_method", "cost_source",
        }},
        "ticker": ticker,
        "market": market,
        "currency": currency,
        "quantity": quantity,
        "average_price": average,
        "cost_method": str(raw.get("cost_method") or "average_cost") if average is not None else None,
        "cost_source": raw.get("cost_source"),
        "diluted_cost": diluted,
        "sellable_quantity": sellable,
    }


def _account(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PortfolioBookError("account must be an object")
    account_id = str(raw.get("account_id") or "").strip()
    if not account_id:
        raise PortfolioBookError("account_id is required")
    currency = str(raw.get("currency") or "USD").strip().upper()
    if currency != "USD":
        raise PortfolioBookError(f"{account_id} currency must be USD for the supported portfolio scope")
    cash = _number(raw.get("cash"), f"{account_id} cash", nullable=True)
    available_cash = _number(raw.get("available_cash"), f"{account_id} available_cash", nullable=True)
    if available_cash is not None and cash is not None and available_cash > cash:
        raise PortfolioBookError(f"{account_id} available_cash cannot exceed cash")
    positions = raw.get("positions", [])
    if not isinstance(positions, list):
        raise PortfolioBookError(f"{account_id} positions must be a list")
    normalized_positions = [_position(p, {"currency": currency}) for p in positions]
    if any(p["market"] != "US" or p["currency"] != "USD" for p in normalized_positions):
        raise PortfolioBookError(f"{account_id} positions must use the supported US/USD scope")
    identities = [(p["ticker"], p["market"], p["currency"]) for p in normalized_positions]
    if len(identities) != len(set(identities)):
        raise PortfolioBookError(f"{account_id} has duplicate positions for the same instrument")
    return {
        **{k: v for k, v in raw.items() if k not in {
            "account_id", "broker", "label", "scope", "currency", "cash", "available_cash",
            "fetched_at", "positions",
        }},
        "account_id": account_id,
        "broker": str(raw.get("broker") or "manual"),
        "label": str(raw.get("label") or account_id),
        "scope": str(raw.get("scope") or "US_STOCKS"),
        "currency": currency,
        "cash": cash,
        "available_cash": available_cash,
        "fetched_at": raw.get("fetched_at"),
        "positions": normalized_positions,
    }


def _legacy_account(book: dict[str, Any]) -> dict[str, Any]:
    currency = str(book.get("currency") or "USD").strip().upper()
    return {
        "account_id": "legacy",
        "broker": "legacy",
        "label": "Legacy portfolio",
        "scope": "US_STOCKS",
        "currency": currency,
        "cash": book.get("cash"),
        # Legacy books only provide a cash balance. They do not establish
        # broker-verified spendable funds, so preserve that distinction.
        "available_cash": None,
        "fetched_at": None,
        "positions": [
            {
                **p,
                "market": p.get("market") or "US",
                "currency": p.get("currency") or currency,
                "cost_method": p.get("cost_method") or "average_cost",
                "cost_source": p.get("cost_source") or "legacy_book",
            }
            for p in (book.get("positions") or [])
        ],
    }


def _projection(accounts: list[dict[str, Any]]) -> tuple[float | None, list[dict[str, Any]]]:
    cash: float | None
    cash_values = [a.get("cash") for a in accounts]
    cash = sum(cash_values) if all(v is not None for v in cash_values) else None
    groups: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for account in accounts:
        for position in account["positions"]:
            identity = (position["ticker"], position["market"], position["currency"])
            groups.setdefault(identity, []).append((account, position))
    rows = []
    for (ticker, market, currency), entries in sorted(groups.items()):
        quantity = sum(p["quantity"] for _, p in entries)
        first = entries[0][1]
        methods = {p.get("cost_method") for _, p in entries}
        cost_known = all(p.get("average_price") is not None for _, p in entries)
        compatible = len(methods) == 1 and None not in methods
        average = (
            sum(p["quantity"] * p["average_price"] for _, p in entries) / quantity
            if cost_known and compatible else None
        )
        rows.append({
            "ticker": ticker,
            "market": market,
            "currency": currency,
            "quantity": quantity,
            "average_price": average,
            "cost_method": next(iter(methods)) if average is not None else None,
            "cost_source": "consolidated" if average is not None else None,
            "sellable_quantity": None,
            "accounts": [a["account_id"] for a, _ in entries],
            "account_positions": [
                {"account_id": a["account_id"], **p} for a, p in entries
            ],
            **{k: first[k] for k in ("name", "type", "sector") if k in first},
        })
    return cash, rows


def normalize_book(book: dict[str, Any] | None) -> dict[str, Any]:
    """Validate/migrate a book in memory and derive the native-compatible view.

    Legacy input is wrapped as a ``legacy`` account without writing it. Callers
    decide when to persist the returned v2 value (normally after preview/apply).
    """
    if book is None:
        book = {}
    if not isinstance(book, dict):
        raise PortfolioBookError("portfolio book must be an object")
    if "schema_version" in book and book.get("schema_version") != 2:
        raise PortfolioBookError(f"unsupported portfolio schema_version: {book.get('schema_version')!r}")
    if "accounts" in book and book.get("schema_version") != 2:
        raise PortfolioBookError("account-based portfolio book requires schema_version 2")
    if book.get("schema_version") == 2:
        raw_accounts = book.get("accounts")
        if not isinstance(raw_accounts, list):
            raise PortfolioBookError("v2 accounts must be a list")
        accounts = [_account(a) for a in raw_accounts]
        ids = [a["account_id"] for a in accounts]
        if len(ids) != len(set(ids)):
            raise PortfolioBookError("account_id values must be unique")
        metadata = {k: v for k, v in book.items() if k not in {
            "schema_version", "accounts", "cash", "currency", "positions",
        }}
    else:
        accounts = [_account(_legacy_account(book))]
        metadata = {k: v for k, v in book.items() if k not in {"cash", "currency", "positions"}}
    cash, positions = _projection(accounts)
    currency = "USD" if all(a["currency"] == "USD" for a in accounts) else None
    return {
        **metadata,
        "schema_version": 2,
        "accounts": accounts,
        "cash": cash,
        "currency": currency,
        "positions": positions,
    }


def replace_account(book: dict[str, Any] | None, account: dict[str, Any], *, replace_legacy: bool = False) -> dict[str, Any]:
    """Replace only the matching broker account and recalculate the projection.

    ``replace_legacy`` explicitly drops the synthetic legacy account when first
    binding a broker; otherwise legacy holdings are preserved as a separate row.
    """
    current = normalize_book(book)
    incoming = _account(account)
    accounts = list(current["accounts"])
    if incoming["account_id"] == "legacy":
        raise PortfolioBookError("legacy is reserved for an existing single-account book")
    if replace_legacy:
        accounts = [a for a in accounts if a["account_id"] != "legacy"]
    found = False
    updated = []
    for existing in accounts:
        if existing["account_id"] == incoming["account_id"]:
            if existing["broker"] != incoming["broker"]:
                raise PortfolioBookError("account_id is already bound to a different broker")
            updated.append(incoming)
            found = True
        else:
            updated.append(existing)
    if not found:
        updated.append(incoming)
    result = normalize_book({**current, "schema_version": 2, "accounts": updated})
    return result
