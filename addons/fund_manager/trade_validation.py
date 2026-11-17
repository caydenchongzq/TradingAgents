"""Pure, deterministic checks for advisory account-local portfolio orders."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
import math
from numbers import Number
from typing import Any

DEFAULT_TRADE_POLICY = {
    "fee_per_order_usd": 3.0,
    "min_buy_notional_usd": 1000.0,
    "min_residual_notional_usd": 1000.0,
}


class TradePolicyError(ValueError):
    """An explicitly supplied owner policy is invalid."""


def _number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, Number):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() else None


def parse_trade_policy(targets: dict | None) -> dict[str, float]:
    if targets is None:
        targets = {}
    if not isinstance(targets, dict):
        raise TradePolicyError("Targets must be an object")
    raw = targets.get("trade_policy", {})
    if not isinstance(raw, dict):
        raise TradePolicyError("trade_policy must be an object")
    result = {}
    for key, default in DEFAULT_TRADE_POLICY.items():
        value = _number(raw[key]) if key in raw else Decimal(str(default))
        if value is None or value < 0 or (key != "fee_per_order_usd" and value == 0):
            raise TradePolicyError(f"trade_policy.{key} must be finite and " +
                                   ("nonnegative" if key == "fee_per_order_usd" else "positive"))
        converted = float(value)
        if not math.isfinite(converted):
            raise TradePolicyError(f"trade_policy.{key} is out of range")
        result[key] = converted
    return result


def _accounts(stats: dict) -> list[dict]:
    if isinstance(stats.get("accounts"), list):
        return deepcopy(stats["accounts"])
    account = stats.get("account") or {}
    return [{"account_id": "default", "cash": account.get("cash"),
             "available_cash": account.get("available_cash"),
             "positions": deepcopy(stats.get("positions") or [])}]


def _float(value: Decimal | None, *, money=False):
    if value is None:
        return None
    return round(float(value), 2) if money else float(value)


def _json_safe(value):
    """Keep invalid proposed numbers readable without emitting invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def validate_trades(decision: dict, stats: dict, policy: dict | None = None) -> dict:
    """Simulate sales before buys without creating orders or changing inputs.

    Validated rows, rather than raw model proposals, own the resulting state.
    Cash availability/settlement remain explicitly hypothetical when not known.
    """
    policy = parse_trade_policy({"trade_policy": policy}) if policy is not None else parse_trade_policy({})
    fee = Decimal(str(policy["fee_per_order_usd"]))
    minimum = Decimal(str(policy["min_buy_notional_usd"]))
    residual = Decimal(str(policy["min_residual_notional_usd"]))
    accounts = _accounts(stats)
    by_id, cash, available, quantities, holdings = {}, {}, {}, {}, {}
    for account in accounts:
        aid = str(account.get("account_id") or "")
        if not aid or aid in by_id:
            raise TradePolicyError("Account context must have unique nonempty IDs")
        by_id[aid] = account
        cash[aid] = _number(account.get("cash"))
        available[aid] = _number(account.get("available_cash"))
        for position in account.get("positions") or []:
            ticker = str(position.get("ticker") or "").upper()
            key = (aid, ticker)
            qty = _number(position.get("quantity"))
            if not ticker or key in holdings or qty is None or qty < 0:
                raise TradePolicyError("Invalid or duplicate account position context")
            holdings[key] = position
            quantities[key] = qty
    initial_cash = dict(cash)
    quotes, known = {}, set()
    for position in (stats.get("positions") or []) + (stats.get("watchlist") or []) + list(holdings.values()):
        ticker = str(position.get("ticker") or "").upper()
        if not ticker:
            continue
        known.add(ticker)
        quote = _number(position.get("price"))
        if ticker not in quotes and quote is not None and quote > 0:
            quotes[ticker] = quote

    proposed_rows = decision.get("trades") or []
    prepared, seen = [], set()
    for index, proposal in enumerate(proposed_rows):
        proposal = proposal if isinstance(proposal, dict) else {}
        ticker = str(proposal.get("ticker") or "").strip().upper()
        aid = str(proposal.get("account_id") or "")
        if not aid and len(by_id) == 1:
            aid = next(iter(by_id))  # compatibility with old single-account reports
        action = str(proposal.get("action") or "HOLD").upper()
        key = (aid, ticker)
        duplicate = key in seen
        seen.add(key)
        prepared.append((index, proposal, aid, ticker, action, duplicate))
    # Reserve duplicate identities in original proposal order, not execution order.
    prepared.sort(key=lambda row: (0 if row[4] == "SELL" else 1 if row[4] == "BUY" else 2, row[0]))
    results, warnings = [], []
    fees = Decimal(0)
    execution_order = 0
    pairs = {"BUY": {"add", "new_position"}, "SELL": {"trim", "exit"}, "HOLD": {"keep", "pass"}}
    for index, proposal, aid, ticker, action, duplicate in prepared:
        key = (aid, ticker)
        requested = _number(proposal.get("shares", 0))
        intent = str(proposal.get("decision") or "").lower()
        result = {**_json_safe(proposal), "ticker": ticker, "account_id": aid, "action": action,
                  "proposal_index": index, "proposed_shares": _json_safe(proposal.get("shares")),
                  "requested_shares": _json_safe(proposal.get("shares")), "effective_shares": 0.0,
                  "shares": 0.0, "fee_usd": 0.0, "value_usd": 0.0,
                  "cash_delta_usd": 0.0, "status": "rejected", "reason": None,
                  "execution_order": None}
        reason, adjustment = None, None
        if duplicate:
            reason = "duplicate account/instrument row; first proposal retained"
        elif aid not in by_id:
            reason = "unknown or missing account"
        elif not ticker or ticker not in known:
            reason = "instrument is not in the book or researched watchlist"
        elif action not in pairs:
            reason = "unsupported action"
        elif intent and intent not in pairs[action]:
            reason = "action and decision are inconsistent"
        elif requested is None or requested < 0:
            reason = "shares must be finite and nonnegative"
        elif action == "HOLD":
            if requested != 0:
                reason = "HOLD requires zero shares"
            else:
                result.update(status="hold", reason="no order")
        elif requested == 0:
            reason = "an actionable order requires positive shares"
        elif str(proposal.get("order_type") or "").lower() == "staged":
            reason = "staged proposals need concrete single orders to estimate fees"
        elif ticker not in quotes:
            reason = "current code-supplied quote unavailable"
        elif cash[aid] is None:
            reason = "account USD cash is unknown; order cannot be validated"
        elif action == "SELL" and key not in holdings:
            reason = "position is not held in this account"
        else:
            px, held = quotes[ticker], quantities.get(key, Decimal(0))
            whole = requested == requested.to_integral_value()
            effective = requested
            source = holdings.get(key, {})
            sellable = _number(source.get("sellable_quantity"))
            limit = min(held, max(sellable, Decimal(0))) if sellable is not None else held
            if action == "BUY":
                if not whole:
                    reason = "buys require whole shares"
                elif initial_cash[aid] < 0:
                    reason = "negative-cash account cannot fund new buys"
            elif intent == "exit":
                effective = limit
                if requested != limit:
                    adjustment = "adjusted to the full available position"
                if limit < held:
                    adjustment = "only available units can be exited; locked residual remains"
            elif requested > held:
                reason = "sell quantity exceeds account holding"
            elif not whole and requested != limit:
                reason = "fractional quantities are allowed only for a complete available-position exit"
            elif requested > limit:
                reason = "sell quantity exceeds verified available units"
            elif 0 < (held - requested) * px < residual:
                effective = limit
                adjustment = ("adjusted to full exit under the residual guideline" if limit == held
                              else "adjusted to all available units; locked residual remains")
            if effective <= 0 and reason is None:
                reason = "no available units to sell"
            gross = effective * px
            delta = (-gross if action == "BUY" else gross) - fee
            if reason is None and action == "BUY" and gross < minimum:
                reason = f"buy value is below ${minimum:,.2f} per-order minimum"
            if reason is None and action == "BUY":
                budget = min(cash[aid], available[aid]) if available[aid] is not None else cash[aid]
                if gross + fee > budget:
                    reason = "insufficient account cash for the purchase and estimated fee"
            if reason is None and cash[aid] + delta < 0:
                reason = "estimated order would leave negative account cash after fees"
            if reason is None:
                execution_order += 1
                cash[aid] += delta
                if available[aid] is not None:
                    available[aid] += delta
                quantities[key] = held + effective if action == "BUY" else held - effective
                if key not in holdings:
                    holdings[key] = {"ticker": ticker, "quantity": 0, "average_price": _float(px),
                                     "price": _float(px), "market": "US", "currency": "USD",
                                     "cost_method": "average_cost", "cost_source": "simulated_entry"}
                elif action == "BUY":
                    old_average = _number(source.get("average_price"))
                    new_average = ((held * old_average + effective * px) / (held + effective)
                                   if old_average is not None else None)
                    holdings[key] = {**source, "average_price": _float(new_average),
                                     "cost_source": "simulated_average"}
                elif sellable is not None:
                    holdings[key] = {**source, "sellable_quantity": _float(min(
                        max(sellable - effective, Decimal(0)), quantities[key]))}
                fees += fee
                caveats = ["hypothetical fill; broker settlement and availability require confirmation"]
                if sellable is None and action == "SELL":
                    caveats.append("sellable quantity was not verified; holding quantity used as the limit")
                if by_id[aid].get("available_cash") is None:
                    caveats.append("broker cash availability was not verified")
                result.update(status="adjusted" if adjustment else "accepted", reason=adjustment,
                              effective_shares=_float(effective), shares=_float(effective),
                              fee_usd=_float(fee, money=True), calc_price=_float(px),
                              value_usd=_float(gross, money=True), cash_delta_usd=_float(delta, money=True),
                              execution_order=execution_order, availability_caveat="; ".join(caveats),
                              fully_closed=action == "SELL" and quantities[key] == 0)
        if reason:
            result["reason"] = reason
            warnings.append(f"{aid or '(missing account)'}/{ticker or '(missing ticker)'}: {reason}")
        results.append(result)
    after = []
    for aid, account in by_id.items():
        positions = []
        for (owner, ticker), source in holdings.items():
            if owner != aid:
                continue
            q = quantities[(owner, ticker)]
            positions.append({**source, "quantity": _float(q)})
        after.append({**account, "cash": _float(cash[aid], money=True),
                      "available_cash": _float(available[aid], money=True), "positions": positions})
    ending = sum(cash.values()) if all(value is not None for value in cash.values()) else None
    return {"trades": results, "warnings": warnings, "accounts_after": after,
            "ending_cash": _float(ending, money=True), "fees_usd": _float(fees, money=True),
            "validated": True, "trade_policy": policy,
            "execution_assumption": "Accepted sells precede buys within the same account; settlement is hypothetical."}
