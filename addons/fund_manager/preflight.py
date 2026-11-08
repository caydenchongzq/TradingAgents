"""Pre-flight for a TradingAgents batch: session date + price-bar sanity.

Runs inside the tradingagents container (TZ=America/New_York) before any LLM
agent is launched. For each ticker it:
  1. checks the cached OHLCV file (data/cache/<SYM>-YFin-data.csv) for a real
     close on the expected session and deletes the file if that bar is
     missing or closeless, so the agents refetch instead of reusing it;
  2. downloads the last ~10 sessions from Yahoo and checks the same bar;
  3. retries the failures after a wait, purging the cache again each time.

Exit codes: 0 ok, 3 a ticker still has no close, 4 market open (live run
refused without --allow-intraday), 2 bad arguments.
Writes <data>/fund_manager/preflight_latest.json and prints KEY=VALUE lines.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from market_calendar import at_ny, expected_bar_for, now_ny, session_info, stamp_ny  # noqa: E402

DATA_ROOT = Path(os.getenv("FM_DATA_ROOT", "/home/appuser/.tradingagents"))


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    from tradingagents.dataflows.symbols import normalize_symbol, safe_ticker_component
    return cache_dir / f"{safe_ticker_component(normalize_symbol(ticker))}-YFin-data.csv"


def _bar_close(df, day: date):
    """(close, last_bar_date) for ``day`` in a frame with a Date column or DatetimeIndex."""
    import pandas as pd
    if df is None or len(df) == 0 or "Close" not in df.columns:
        return None, None
    # Yahoo daily stamps are midnight in the exchange zone, and a cache can mix
    # -05:00 and -04:00. The session date is the first 10 characters. Parsing the
    # whole column with pandas raises "Mixed timezones" and would delete a good file.
    if "Date" in df.columns:
        dates = []
        for value in df["Date"].tolist():
            text = str(value)
            try:
                dates.append(date.fromisoformat(text[:10]))
            except ValueError:
                dates.append(None)
    else:
        dates = list(pd.DatetimeIndex(df.index).date)
    last = dates[-1] if dates else None
    for i in range(len(dates) - 1, -1, -1):
        if dates[i] is None:
            continue
        if dates[i] == day:
            v = df["Close"].iloc[i]
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None, last
            return (v if math.isfinite(v) and v > 0 else None), last
        if dates[i] < day:
            break
    return None, last


def _settled_at(day: date) -> datetime:
    from datetime import timedelta
    from market_calendar import SETTLE_MINUTES, close_time
    return at_ny(day, close_time(day)) + timedelta(minutes=SETTLE_MINUTES)


def check_cache(cache_dir: Path, ticker: str, day: date, purge: bool) -> str:
    import pandas as pd
    path = _cache_path(cache_dir, ticker)
    if not path.exists():
        return "absent"
    written = stamp_ny(path.stat().st_mtime)
    if written < _settled_at(day) and written.date() >= day:
        # Downloaded while the expected session was still trading: its bar is partial.
        if purge:
            path.unlink(missing_ok=True)
            return f"purged (written {written:%Y-%m-%d %H:%M} ET, before the {day} bar settled)"
        return "bad (partial bar)"
    try:
        df = pd.read_csv(path, on_bad_lines="skip", encoding="utf-8")
        close, last = _bar_close(df, day)
    except Exception as e:  # unreadable file is as bad as a missing bar
        close, last = None, f"unreadable: {e}"
    if close is not None:
        return "ok"
    if purge:
        path.unlink(missing_ok=True)
        return f"purged (bar {day} {'closeless' if last and str(last) >= str(day) else 'missing'}; last row {last})"
    return f"bad (last row {last})"


def fetch_bar(ticker: str, day: date):
    import yfinance as yf
    from tradingagents.dataflows.symbols import normalize_symbol
    try:
        hist = yf.Ticker(normalize_symbol(ticker)).history(period="10d", auto_adjust=True, actions=False)
    except Exception as e:
        return None, None, f"download error: {e}"
    close, last = _bar_close(hist, day)
    if close is None:
        return None, last, "no bars" if last is None else f"no final close for {day} (last bar {last})"
    return close, last, ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="", help="comma list")
    ap.add_argument("--analysis-date", default="auto", help="auto = New York date (live mode)")
    ap.add_argument("--allow-intraday", action="store_true")
    ap.add_argument("--block-intraday", action="store_true",
                    help="exit 4 when the cash session is open or still settling (live agent runs)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--wait", type=int, default=120, help="seconds between retries")
    ap.add_argument("--cache-dir", default=str(DATA_ROOT / "cache"))
    ap.add_argument("--out", default=str(DATA_ROOT / "fund_manager" / "preflight_latest.json"))
    ap.add_argument("--session-only", action="store_true", help="print the session/date decision and exit")
    a = ap.parse_args(argv)

    info = session_info()
    if a.analysis_date == "auto":
        analysis = date.fromisoformat(info["analysis_date"])
    else:
        try:
            analysis = date.fromisoformat(a.analysis_date)
        except ValueError:
            print(f"PREFLIGHT_STATUS=bad_args\nbad --analysis-date {a.analysis_date!r}")
            return 2
    ny_today = date.fromisoformat(info["ny_date"])
    mode = "live" if analysis >= ny_today else "historical"
    expected = expected_bar_for(analysis)
    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]

    result = {
        "generated_ny": now_ny().isoformat(timespec="seconds"),
        "session": info,
        "analysis_date": analysis.isoformat(),
        "mode": mode,
        "expected_session": expected.isoformat(),
        "tickers": {},
        "failed": [],
        "warnings": [],
    }
    if mode == "historical":
        result["warnings"].append(
            f"analysis date {analysis} is before NY today {ny_today}: TradingAgents runs in historical mode "
            "(live profile, insider trades, Polymarket withheld; news/StockTwits capped at that date)")
    if analysis > ny_today:
        result["warnings"].append(f"analysis date {analysis} is after NY today {ny_today}")

    status = "ok"
    intraday_block = mode == "live" and info["intraday"] and not a.allow_intraday
    if intraday_block:
        result["warnings"].append(
            f"US market is {info['market_state']} ({info['ny_now']}); a live agent run now would read a partial bar. "
            "Wait for the close (+30 min) or pass -AllowIntraday.")
        if a.block_intraday:
            status = "blocked_intraday"
    if status != "blocked_intraday" and not a.session_only and tickers:
        cache_dir = Path(a.cache_dir)
        pending = list(tickers)
        for t in tickers:
            result["tickers"][t] = {"cache": check_cache(cache_dir, t, expected, purge=True)}
        for attempt in range(a.retries + 1):
            still = []
            for t in pending:
                close, last, issue = fetch_bar(t, expected)
                result["tickers"][t].update({"close": close, "last_bar": str(last) if last else None,
                                             "issue": issue, "attempts": attempt + 1})
                if close is None:
                    still.append(t)
                    result["tickers"][t]["cache_retry"] = check_cache(cache_dir, t, expected, purge=True)
            pending = still
            if not pending or attempt == a.retries:
                break
            print(f"[preflight] {len(pending)} ticker(s) without a final {expected} close: {', '.join(pending)}; "
                  f"retry {attempt + 1}/{a.retries} in {a.wait}s", flush=True)
            time.sleep(a.wait)
        result["failed"] = pending
        if pending:
            status = "failed"
    result["status"] = status

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for w in result["warnings"]:
        print(f"[preflight] WARNING: {w}")
    for t, r in result["tickers"].items():
        mark = "OK " if r.get("close") is not None else "BAD"
        print(f"[preflight] {mark} {t:<6} {expected} close={r.get('close')} cache={r.get('cache')} {r.get('issue') or ''}".rstrip())
    print(f"MARKET_STATE={info['market_state']}")
    print(f"NY_NOW={info['ny_now']}")
    print(f"ANALYSIS_DATE={analysis.isoformat()}")
    print(f"EXPECTED_SESSION={expected.isoformat()}")
    print(f"MODE={mode}")
    print(f"PREFLIGHT_STATUS={status}")
    if result["failed"]:
        print(f"FAILED_TICKERS={','.join(result['failed'])}")
    return {"ok": 0, "failed": 3, "blocked_intraday": 4}[status]


if __name__ == "__main__":
    sys.exit(main())
