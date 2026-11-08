"""Point-in-time macro, prices, and headlines for the theme scout and fund manager.

Does not edit TradingAgents. FRED observations use the same vintage pin as
``tradingagents.dataflows.vendors.fred.get_macro_data``: ``realtime_start`` and
``realtime_end`` are the decision date, clamped to FRED's Chicago today.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

FRED_SERIES = {
    "fed_funds": "FEDFUNDS",
    "10y_treasury": "DGS10",
    "2y_treasury": "DGS2",
    "10y_2y_spread": "T10Y2Y",
    "vix": "VIXCLS",
    "core_cpi_index": "CPILFESL",
    "unemployment": "UNRATE",
    "dollar_index": "DTWEXBGS",
}
FRED_MONTHLY = {"fed_funds", "core_cpi_index", "unemployment"}
FRED_LOOKBACK_DAYS = 800
STALE_DAILY_DAYS = 4
STALE_MONTHLY_DAYS = 45


def fred_observation_params(as_of: str, series_id: str, start: str, fred_today: str) -> dict:
    """Parameters for one vintage-pinned FRED observations request.

    ``observation_end`` stays on the decision date. The realtime window is the
    vintage that was published by that date, and it cannot sit after FRED's own
    today or the request is rejected.
    """
    pit = min(as_of, fred_today)
    return {
        "series_id": series_id,
        "observation_start": start,
        "observation_end": as_of,
        "sort_order": "desc",
        "limit": 900,
        "realtime_start": pit,
        "realtime_end": pit,
    }


def series_from_observations(name: str, series_id: str, observations: list, as_of: str, vintage: str) -> dict:
    """One macro field. ``as_of`` is the decision date. ``date`` is the print date."""
    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")
    points = [o for o in observations or [] if str(o.get("value")) not in {".", "None", ""} and o.get("value") is not None]
    if not points:
        return {"as_of": as_of, "vintage": vintage, "stale": True, "error": "no observations"}
    latest = points[0]
    obs_dt = datetime.strptime(latest["date"], "%Y-%m-%d")
    age = (as_of_dt - obs_dt).days
    limit = STALE_MONTHLY_DAYS if name in FRED_MONTHLY else STALE_DAILY_DAYS
    entry = {
        "as_of": as_of,
        "vintage": vintage,
        "date": latest["date"],
        "value": float(latest["value"]),
        "age_days": age,
        "stale": age > limit,
    }
    target = (obs_dt - timedelta(days=365)).strftime("%Y-%m-%d")
    older = [o for o in points if o["date"] <= target]
    if older:
        entry["value_1y_ago"] = float(older[0]["value"])
        if series_id == "CPILFESL" and entry["value_1y_ago"]:
            entry["yoy_pct"] = round((entry["value"] / entry["value_1y_ago"] - 1) * 100, 2)
    return entry


def fred_snapshot(as_of: str, series: dict | None = None, *, fred=None, fred_today: str | None = None) -> dict:
    """Latest vintage-pinned FRED values. A missing vendor returns as_of and stale."""
    series = series or FRED_SERIES
    if fred is None:
        try:
            from tradingagents.dataflows.vendors import fred as fred_mod
            fred = fred_mod
        except Exception as exc:  # noqa: BLE001
            return {"error": _safe(exc), "as_of": as_of, "stale": True}
    if fred_today is None:
        try:
            fred_today = fred._fred_today()
        except Exception:  # noqa: BLE001
            fred_today = as_of
    vintage = min(as_of, fred_today)
    start = (datetime.strptime(as_of, "%Y-%m-%d") - timedelta(days=FRED_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    out = {"as_of": as_of, "vintage": vintage}
    for name, sid in series.items():
        params = fred_observation_params(as_of, sid, start, fred_today)
        try:
            data = fred._request("series/observations", params)
            obs = data.get("observations", []) if isinstance(data, dict) else []
            out[name] = series_from_observations(name, sid, obs, as_of, vintage)
        except Exception as exc:  # noqa: BLE001
            out[name] = {"as_of": as_of, "vintage": vintage, "stale": True, "error": _safe(exc)}
    return out


def history_window(as_of: str, days: int) -> tuple[str, str]:
    """``start`` and ``end`` for a yfinance download that stops at the decision date.

    yfinance treats ``end`` as exclusive, so it is the day after ``as_of``.
    """
    day = datetime.strptime(as_of, "%Y-%m-%d")
    return (day - timedelta(days=days)).strftime("%Y-%m-%d"), (day + timedelta(days=1)).strftime("%Y-%m-%d")


def clip_frame(frame, as_of: str | None):
    """Drop price bars after the decision date. ``period=`` downloads end at the wall clock."""
    if frame is None or as_of is None or getattr(frame, "empty", True):
        return frame
    import pandas as pd

    out = frame.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    return out.loc[out.index <= pd.Timestamp(as_of)]


def headline_day(value) -> str | None:
    """Calendar day from an ISO stamp, a unix time, or an Alpha Vantage ``YYYYMMDDThhmmss`` stamp."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), timezone.utc).strftime("%Y-%m-%d")
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if re.fullmatch(r"\d{8}T\d{6}", text):
        return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"
    match = re.match(r"(\d{4}-\d{2}-\d{2})", text)
    if match:
        return match.group(1)
    return None


def filter_headlines(rows: list, as_of: str) -> list:
    """Keep headlines on or before ``as_of``. A later print is lookahead. Undated rows stay and are marked."""
    kept = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        day = headline_day(item.get("date") or item.get("published") or item.get("time_published"))
        if day and day > as_of:
            continue
        item["as_of"] = as_of
        if item.get("timing") is None:
            item["timing"] = "undated" if day is None and not item.get("error") else "as_of"
        if day:
            item["headline_date"] = day
        kept.append(item)
    return kept


def mark_live(row: dict, *keys: str) -> dict:
    """Yahoo earnings calendars and quote snapshots have no historical vintage."""
    out = dict(row)
    out["timing"] = "live"
    out["live_fields"] = list(keys) or out.get("live_fields") or []
    return out


def _safe(err) -> str:
    text = re.sub(r"(?i)(apikey|api_key|token)=[^&\s]+", r"\1=***", str(err))
    return text[:180]
