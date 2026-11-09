#!/usr/bin/env python3
"""Weekly theme scout and daily quant screen. Does not modify TradingAgents source.

--screen   free ranked screen only
--no-llm   weekly pack (macro, headlines, earnings, relative strength, screen) and no model call
--dry-run  same as --no-llm: no model call and no watchlist write

A weekly run without those flags makes one deep-model call and then updates the watchlist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(os.getenv("FM_DATA_ROOT", "/home/appuser/.tradingagents"))
DEEP_MODEL = os.getenv("THEME_SCOUT_MODEL", "openai/gpt-6.1-sol")
OPENROUTER_BASE = os.getenv("FUND_MANAGER_BASE_URL", "https://openrouter.ai/api/v1")
MIN_WATCH = 0  # retained for old callers; no watchlist floor is enforced
MAX_WATCH = 10_000  # retained for old callers; candidates are never auto-trimmed
SECTORS = ["XLK", "XLY", "XLV", "XLP", "XLU", "XLE", "XLF", "SMH"]
WINDOWS = {"1w": 5, "1m": 21, "3m": 63}
HISTORY_DAYS = 430  # about 14 months of calendar days
MAX_DISCOVERY_PER_SECTOR = 20

SECTOR_NAMES = {
    "XLK": "Technology", "XLY": "Consumer Cyclical", "XLV": "Healthcare",
    "XLP": "Consumer Defensive", "XLU": "Utilities", "XLE": "Energy",
    "XLF": "Financial Services", "SMH": "Semiconductors",
}
SECTOR_INDUSTRIES = {"SMH": "Semiconductors"}
YAHOO_SECTOR_NAMES = {"SMH": "Technology"}


def discover_sector_candidates(sectors: list[str], per_sector: int = MAX_DISCOVERY_PER_SECTOR) -> list[dict]:
    """Fetch a bounded, liquid US stock pool from Yahoo's sector screener."""
    import yfinance as yf
    from yfinance import EquityQuery

    rows, seen = [], set()
    for sector in sectors:
        label = SECTOR_NAMES[sector]
        yahoo_sector = YAHOO_SECTOR_NAMES.get(sector, label)
        try:
            parts = [
                EquityQuery("eq", ["region", "us"]),
                EquityQuery("eq", ["sector", yahoo_sector]),
                EquityQuery("gt", ["avgdailyvol3m", 250000]),
                EquityQuery("gt", ["intradaymarketcap", 500000000]),
                EquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
            ]
            if sector in SECTOR_INDUSTRIES:
                parts.append(EquityQuery("eq", ["industry", SECTOR_INDUSTRIES[sector]]))
            query = EquityQuery("and", parts)
            result = yf.screen(query, size=min(max(int(per_sector), 1), 50),
                               sortField="avgdailyvol3m", sortAsc=False)
        except Exception as exc:  # a sector feed outage should not invent candidates
            log(f"Yahoo discovery unavailable for {sector}: {_safe(exc)}")
            continue
        quotes = (result or {}).get("quotes") or []
        added = 0
        for quote in quotes:
            if not isinstance(quote, dict) or str(quote.get("quoteType") or "").upper() not in {"EQUITY", "STOCK"}:
                continue
            ticker = str(quote.get("symbol") or "").strip().upper()
            if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", ticker) or ticker in seen:
                continue
            seen.add(ticker)
            rows.append({"ticker": ticker, "sector": label, "type": "stock",
                         "discovery_source": "Yahoo Finance screener"})
            added += 1
            if added >= per_sector:
                break
    return rows


def log(msg: str) -> None:
    print(f"[theme-scout] {msg}", flush=True)


def _safe(err) -> str:
    text = re.sub(r"(?i)(apikey|api_key|token)=[^&\s]+", r"\1=***", str(err))
    return text[:180]


def load_json(path: Path, default=None):
    if path and path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def _themes_path(explicit: str) -> Path | None:
    if explicit:
        return Path(explicit)
    candidates = [
        Path("/home/appuser/addon/themes.json"),
        Path(__file__).resolve().parents[1] / "fund_manager" / "themes.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_watchlist(path: Path) -> list[dict]:
    raw = load_json(path, [])
    if isinstance(raw, dict):
        raw = next((raw.get(key) for key in ("items", "candidates", "watchlist", "tickers")
                    if isinstance(raw.get(key), list)), [])
    out, seen = [], set()
    for item in raw or []:
        if isinstance(item, str):
            item = {"ticker": item}
        if not isinstance(item, dict) or not item.get("ticker"):
            continue
        tk = str(item["ticker"]).strip().upper()
        if tk in seen:
            continue
        seen.add(tk)
        row = {"ticker": tk, "sector": item.get("sector"), "type": item.get("type") or "stock"}
        if item.get("theme"):
            row["theme"] = item.get("theme")
        if item.get("sub_theme"):
            row["sub_theme"] = item.get("sub_theme")
        if item.get("note"):
            row["note"] = item.get("note")
        out.append(row)
    return out


def watchlist_version(items: list[dict]) -> str:
    """Stable hash of the candidate fields consumed by the scout."""
    rows = []
    for item in items:
        ticker = str(item.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        row = {"ticker": ticker, "sector": item.get("sector"), "type": item.get("type") or "stock"}
        for key in ("theme", "sub_theme", "note"):
            if item.get(key):
                row[key] = item[key]
        rows.append(row)
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _panel(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if field in raw.columns.get_level_values(0):
            frame = raw[field]
        elif field in raw.columns.get_level_values(1):
            frame = raw.xs(field, axis=1, level=1)
        else:
            return pd.DataFrame()
    else:
        if field not in raw:
            return pd.DataFrame()
        frame = raw[[field]] if not isinstance(raw[field], pd.DataFrame) else raw[field]
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    frame = frame.copy()
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index = pd.to_datetime(frame.index).normalize()
    return frame.sort_index()


def _addons():
    root = Path(__file__).resolve().parents[1]
    if not (root / "macro_panel.py").is_file():
        root = Path("/home/appuser/addons")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def download_ohlc(tickers: list[str], as_of: str | None = None) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    _addons()
    import macro_panel

    symbols = [t for t in dict.fromkeys(tickers) if t]
    if as_of:
        # A fixed window that ends at the decision date. period= would end at the wall clock.
        start, end = macro_panel.history_window(as_of, HISTORY_DAYS)
        raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False, threads=False,
                          group_by="column")
    else:
        raw = yf.download(symbols, period="14mo", auto_adjust=True, progress=False, threads=False, group_by="column")
    panels = {field: _panel(raw, field) for field in ("Open", "High", "Low", "Close")}
    if as_of:
        panels = {field: macro_panel.clip_frame(frame, as_of) for field, frame in panels.items()}
    return panels


def _ret(series: pd.Series, sessions: int) -> float | None:
    s = series.dropna()
    if len(s) <= sessions:
        return None
    base = float(s.iloc[-1 - sessions])
    if not base:
        return None
    return float(s.iloc[-1] / base - 1)


def _ma_dist(series: pd.Series, window: int) -> float | None:
    s = series.dropna()
    if len(s) < window:
        return None
    ma = float(s.tail(window).mean())
    if not ma:
        return None
    return float(s.iloc[-1] / ma - 1) * 100


def _atr_pct(high: pd.Series, low: pd.Series, close: pd.Series) -> float | None:
    if high is None or low is None or close is None:
        return None
    idx = high.dropna().index.intersection(low.dropna().index).intersection(close.dropna().index)
    if len(idx) < 15:
        return None
    hh, ll, cc = high.loc[idx], low.loc[idx], close.loc[idx]
    prev = cc.shift(1)
    tr = pd.concat([(hh - ll), (hh - prev).abs(), (ll - prev).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    last = float(cc.iloc[-1])
    if not last or not math.isfinite(atr):
        return None
    return atr / last * 100


def _basket_ret(close: pd.DataFrame, members: list[str], sessions: int) -> float | None:
    rets = []
    for name in members:
        if name not in close:
            continue
        value = _ret(close[name], sessions)
        if value is not None and math.isfinite(value):
            rets.append(value)
    if not rets:
        return None
    return float(sum(rets) / len(rets))


def _z(values: list[float | None]) -> list[float]:
    clean = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if len(clean) < 2:
        return [0.0 for _ in values]
    mu = float(np.mean(clean))
    sd = float(np.std(clean))
    if sd == 0:
        return [0.0 for _ in values]
    out = []
    for v in values:
        if isinstance(v, (int, float)) and math.isfinite(v):
            out.append((float(v) - mu) / sd)
        else:
            out.append(0.0)
    return out


def earnings_days(ticker: str, today: date) -> int | None:
    import yfinance as yf

    try:
        frame = yf.Ticker(ticker).get_earnings_dates(limit=8)
        index = getattr(frame, "index", None)
        if frame is None or index is None or len(index) == 0:
            return None
        future = []
        for idx in list(index):
            raw = idx.date() if hasattr(idx, "date") else None
            if isinstance(raw, datetime):
                raw = raw.date()
            if isinstance(raw, date) and raw >= today:
                future.append(raw)
        if not future:
            return None
        return (min(future) - today).days
    except Exception:  # noqa: BLE001
        return None


def screen_frame(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                 tickers: list[str], spy: str, earnings: dict[str, int | None]) -> list[dict]:
    rows = []
    spy_rets = {label: _ret(close[spy], n) if spy in close else None for label, n in WINDOWS.items()}
    for tk in tickers:
        if tk not in close:
            rows.append({"ticker": tk, "error": "no prices"})
            continue
        series = close[tk]
        row = {"ticker": tk, "price": round(float(series.dropna().iloc[-1]), 4) if series.dropna().size else None}
        for label, n in WINDOWS.items():
            ret = _ret(series, n)
            row[f"ret_{label}"] = None if ret is None else round(ret * 100, 2)
            spy_ret = spy_rets.get(label)
            row[f"rs_{label}"] = None if ret is None or spy_ret is None else round((ret - spy_ret) * 100, 2)
        for window in (20, 50, 200):
            dist = _ma_dist(series, window)
            row[f"dist_ma{window}"] = None if dist is None else round(dist, 2)
        h = high[tk] if tk in high else None
        l = low[tk] if tk in low else None
        atr = _atr_pct(h, l, series)
        row["atr_pct"] = None if atr is None else round(atr, 2)
        days = earnings.get(tk)
        row["earnings_days"] = days
        # Yahoo's earnings calendar is a live feed. It has no vintage for --date.
        row["earnings_timing"] = "live"
        rows.append(row)
    zs = {key: _z([r.get(key) for r in rows]) for key in ("rs_1w", "rs_1m", "rs_3m")}
    for i, row in enumerate(rows):
        if row.get("error"):
            row["score"] = None
            continue
        above50 = 1.0 if (row.get("dist_ma50") or 0) > 0 else -1.0
        above200 = 1.0 if (row.get("dist_ma200") or 0) > 0 else 0.0
        penalty = 0.5 if isinstance(row.get("earnings_days"), int) and row["earnings_days"] <= 7 else 0.0
        row["score"] = round(0.20 * zs["rs_1w"][i] + 0.40 * zs["rs_1m"][i] + 0.25 * zs["rs_3m"][i]
                             + 0.10 * above50 + 0.05 * above200 - penalty, 3)
    ranked = sorted(rows, key=lambda r: (-(r["score"] if r.get("score") is not None else -999), r["ticker"]))
    for i, row in enumerate(ranked, 1):
        row["rank"] = i
    return ranked


def group_rs(close: pd.DataFrame, groups: dict[str, list[str]], spy: str) -> list[dict]:
    out = []
    spy_rets = {label: _ret(close[spy], n) if spy in close else None for label, n in WINDOWS.items()}
    for name, members in groups.items():
        row = {"id": name, "members": len(members)}
        for label, n in WINDOWS.items():
            basket = _basket_ret(close, members, n)
            spy_ret = spy_rets.get(label)
            row[f"ret_{label}"] = None if basket is None else round(basket * 100, 2)
            row[f"rs_{label}"] = None if basket is None or spy_ret is None else round((basket - spy_ret) * 100, 2)
        out.append(row)
    return out


def fred_snapshot(as_of: str) -> dict:
    _addons()
    import macro_panel
    return macro_panel.fred_snapshot(as_of)


def yahoo_headlines(symbols: list[str], limit: int = 12) -> list[dict]:
    import yfinance as yf

    rows = []
    for tk in symbols:
        try:
            articles = yf.Ticker(tk).news or []
        except Exception as e:  # noqa: BLE001
            rows.append({"source": "yfinance", "ticker": tk, "error": _safe(e)})
            continue
        for article in articles:
            content = article.get("content") if isinstance(article, dict) else None
            if isinstance(content, dict):
                title = content.get("title")
                when = content.get("pubDate")
            else:
                title = article.get("title") if isinstance(article, dict) else None
                when = None
            if title:
                rows.append({"source": "yfinance", "ticker": tk, "title": title, "date": when})
            if len(rows) >= limit:
                return rows
    return rows


def av_headlines(as_of: str, limit: int = 12) -> list[dict]:
    try:
        from tradingagents.dataflows.vendors.alpha_vantage.news import get_global_news
        raw = get_global_news(as_of, look_back_days=7, limit=limit)
    except Exception as e:  # noqa: BLE001
        return [{"source": "alpha_vantage", "error": _safe(e)}]
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return [{"source": "alpha_vantage", "error": _safe(raw)}]
    feed = raw.get("feed") if isinstance(raw, dict) else None
    if not feed:
        return [{"source": "alpha_vantage", "error": "no feed"}]
    return [{"source": "alpha_vantage", "title": item.get("title"), "date": item.get("time_published"),
             "summary": (item.get("summary") or "")[:240]} for item in feed[:limit]]


MAX_ADDS_PER_WEEK = None
MAX_DROPS_PER_WEEK = None
COOLDOWN_DAYS = 14


def curator_policy(targets: dict | None) -> dict:
    return {
        # Weekly activity caps are deliberately unlimited. Keep reading the
        # old fields nowhere in enforcement so legacy values cannot restrict
        # novice experimentation.
        "max_adds_per_week": None,
        "max_drops_per_week": None,
        "cooldown_days": COOLDOWN_DAYS,
    }


def staged_tickers(data_root: Path) -> set[str]:
    """Tickers with a staged order in the newest fund-manager report."""
    base = Path(data_root) / "fund_manager"
    if not base.is_dir():
        return set()
    reports = [path for path in base.glob("*/*/portfolio_report.json") if path.is_file()]
    if not reports:
        return set()
    latest = max(reports, key=lambda path: path.stat().st_mtime)
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    names = set()
    for key in ("final", "draft"):
        for trade in (data.get(key) or {}).get("trades") or []:
            if str(trade.get("order_type") or "").lower() != "staged":
                continue
            try:
                shares = int(trade.get("shares") or 0)
            except (TypeError, ValueError):
                shares = 0
            ticker = str(trade.get("ticker") or "").upper()
            if ticker and shares:
                names.add(ticker)
    return names


def curator_shadow(as_of: str | None, targets: dict | None = None) -> bool:
    """The scout always stages proposed changes for a separate review."""
    return True


def ai_weight_for(ticker: str, targets: dict | None, themes: dict | None) -> tuple[float | None, str]:
    """Return an explicit AI weight; leave untagged names unclassified."""
    sleeve = (targets or {}).get("ai_sleeve") or {}
    ticker_weight = sleeve.get("ticker_weight") or {}
    if ticker in ticker_weight:
        return float(ticker_weight[ticker]), "ticker_weight"
    theme_weight = sleeve.get("theme_weight") or {}
    for name, members in ((themes or {}).get("themes") or {}).items():
        if ticker in {str(member).upper() for member in members}:
            if name in theme_weight:
                return float(theme_weight[name]), "theme_weight"
            # Tagged under a theme the AI sleeve does not count, such as sector_etf.
            return 0.0, f"theme {name}"
    return None, "unclassified"


def ranked_watchlist_additions(ranking: list[dict], proposed_adds: list[dict],
                               candidates: dict[str, dict], holdings: set[str]
                               ) -> tuple[list[dict], list[str]]:
    """Stage only validated candidates ranked add; ranking is the source of truth.

    The schema has both a ranking action and a separate add array. Use the array
    for optional metadata, but don't let disagreement silently hide a ranked
    candidate or bypass screening and held-position checks.
    """
    explicit = {}
    for item in proposed_adds or []:
        ticker = str(item.get("ticker") or "").strip().upper()
        if ticker:
            explicit.setdefault(ticker, item)

    ranked = {}
    held = {str(ticker).upper() for ticker in holdings}
    for row in ranking or []:
        ticker = str(row.get("ticker") or "").strip().upper()
        if (row.get("bucket") == "candidate" and row.get("action") == "add"
                and not row.get("action_conflict") and ticker):
            ranked.setdefault(ticker, row)

    additions, mismatches = [], []
    for ticker, row in ranked.items():
        validated = candidates.get(ticker)
        if ticker in held or not validated:
            mismatches.append(f"ranking add {ticker} is held or outside screened candidate universe")
            continue
        details = explicit.get(ticker) or {}
        additions.append({
            **details,
            "ticker": ticker,
            "sector": validated.get("sector") or details.get("sector") or "Unclassified",
            "type": validated.get("type") or "stock",
            "reason": details.get("reason") or row.get("reason") or "worth monitoring",
        })

    for ticker in explicit.keys() - ranked.keys():
        mismatches.append(f"add array includes {ticker}, but ranking did not mark it add")
    return additions, mismatches


def ranked_watchlist_drops(ranking: list[dict], proposed_drops: list[dict]) -> tuple[list[dict], list[str]]:
    """Accept drops only for consistently ranked candidates with a negative action."""
    by_ticker = {str(row.get("ticker") or "").strip().upper(): row
                 for row in ranking or [] if isinstance(row, dict)}
    drops, warnings = [], []
    for item in proposed_drops or []:
        if not isinstance(item, dict):
            warnings.append("Ignored malformed drop recommendation.")
            continue
        ticker = str(item.get("ticker") or "").strip().upper()
        row = by_ticker.get(ticker)
        if (row and row.get("bucket") == "candidate"
                and row.get("action") in {"pass", "trim"}
                and not row.get("action_conflict")):
            drops.append(item)
        else:
            warnings.append(f"Ignored drop recommendation for {ticker or 'unnamed ticker'} without a consistent negative candidate action.")
    return drops, warnings


def book_ai_lookthrough(book: dict | None, close: pd.DataFrame, targets: dict | None,
                        themes: dict | None) -> float | None:
    """AI look-through % of the current book at the as-of close. Same weights as the fund manager.

    A held name with no tag counts 0 here, as in the fund manager. Only new adds default to 1.0.
    """
    positions = (book or {}).get("positions") or []
    cash = float((book or {}).get("cash") or 0)
    sleeve = (targets or {}).get("ai_sleeve") or {}
    tick = sleeve.get("ticker_weight") or {}
    theme_w = sleeve.get("theme_weight") or {}
    invested, ai = 0.0, 0.0
    for pos in positions:
        tk = str(pos.get("ticker") or "").upper()
        qty = float(pos.get("quantity") or pos.get("shares") or 0)
        if not tk or not qty:
            continue
        px = None
        if close is not None and tk in close:
            series = close[tk].dropna()
            if len(series):
                px = float(series.iloc[-1])
        if px is None:
            px = float(pos.get("average_price") or 0)
        value = qty * px
        invested += value
        if tk in tick:
            frac = float(tick[tk])
        else:
            frac = 0.0
            for name, members in ((themes or {}).get("themes") or {}).items():
                if tk in {str(m).upper() for m in members}:
                    frac = float(theme_w.get(name, 0.0))
                    break
        ai += value * frac
    total = invested + cash
    if total <= 0:
        return None
    return round(ai / total * 100, 2)


def apply_watchlist(items: list[dict], adds: list[dict], drops: list[dict],
                    scores: dict[str, float], min_n: int = MIN_WATCH, max_n: int = MAX_WATCH, *,
                    shadow: bool | None = None, as_of: str | None = None,
                    holdings: set | None = None, staged: set | None = None,
                    universe: set | None = None, log_events: list | None = None,
                    targets: dict | None = None, themes: dict | None = None,
                    ai_lookthrough_pct: float | None = None) -> tuple[list[dict], list[dict]]:
    """Build a proposed watchlist while preserving protected holdings and staged orders."""
    # Keep the legacy keyword for callers, but scouting can no longer authorize
    # a live watchlist edit. Applying a proposal belongs to the reviewed UI flow.
    policy = curator_policy(targets)
    held = {str(tk).upper() for tk in (holdings or set())}
    staged_names = {str(tk).upper() for tk in (staged or set())}
    allowed = {str(tk).upper() for tk in universe} if universe is not None else None
    _addons()
    import watchlist_log
    events = list(log_events or [])
    day = as_of or date.today().isoformat()
    cooling = watchlist_log.cooldown_tickers(events, day, policy["cooldown_days"])
    by, order, changes = {}, [], []
    for item in items:
        tk = str(item["ticker"]).upper()
        if tk in by:
            continue
        row = dict(item)
        row["ticker"] = tk
        by[tk] = row
        order.append(tk)
    fresh = set()

    def _change(action: str, ticker: str, reason: str, **extra) -> None:
        row = {"action": action, "ticker": ticker, "reason": reason, "shadow": bool(shadow), "applied": False}
        row.update(extra)
        changes.append(row)

    for add in adds or []:
        tk = str(add.get("ticker") or "").strip().upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", tk or ""):
            _change("skip_add", tk, "invalid ticker")
            continue
        if allowed is not None and tk not in allowed:
            _change("skip_add", tk, "outside the universe")
            continue
        if tk in by:
            _change("skip_add", tk, "already on the watchlist")
            continue
        if tk in cooling:
            _change("skip_add", tk, f"cooldown of {policy['cooldown_days']} days after an add or drop")
            continue
        weight, source = ai_weight_for(tk, targets, themes)
        # A watchlist addition means "monitor this idea", not "buy it". Current
        # portfolio exposure belongs in portfolio review and must not hide ideas
        # from the separately approved monitoring list.
        row = {"ticker": tk, "sector": add.get("sector") or "Unclassified", "type": add.get("type") or "stock",
               "ai_weight": weight, "ai_weight_source": source}
        if add.get("sub_theme"):
            row["sub_theme"] = add["sub_theme"]
        if add.get("reason"):
            row["note"] = add["reason"]
        by[tk] = row
        order.append(tk)
        fresh.add(tk)
        _change("add", tk, add.get("reason") or "", ai_weight=weight, ai_weight_source=source)

    def _protected(tk: str) -> str | None:
        if tk in held:
            return "holding is never dropped; flagged trim"
        if tk in staged_names:
            return "staged order is never dropped; flagged trim"
        return None

    for drop in drops or []:
        tk = str(drop.get("ticker") if isinstance(drop, dict) else drop or "").strip().upper()
        reason = drop.get("reason") if isinstance(drop, dict) else ""
        if tk not in by:
            _change("skip_drop", tk, "not on the watchlist")
            continue
        why = _protected(tk)
        if why:
            _change("trim", tk, why if not reason else f"{why}. {reason}")
            continue
        if tk in cooling:
            _change("skip_drop", tk, f"cooldown of {policy['cooldown_days']} days after an add or drop")
            continue
        by.pop(tk)
        order = [t for t in order if t != tk]
        _change("drop", tk, reason)
    proposed = [by[tk] for tk in order if tk in by]
    _change("shadow", "", "proposals only; watchlist not edited")
    return proposed, changes


def commit_watchlist(path: Path, current: list[dict], proposed: list[dict], changes: list[dict],
                     proposals_path: Path, data_root: Path, as_of: str,
                     sectors: list[str] | None = None, coverage: dict | None = None) -> str:
    """Persist a reviewable proposal; only the explicit watchlist flow may apply it."""
    proposals_path.parent.mkdir(parents=True, exist_ok=True)
    staged_changes = [{**change, "shadow": True, "applied": False} for change in changes]
    payload = {"date": as_of, "shadow": True, "changes": staged_changes, "proposed": proposed,
               "watchlist_size": len(current), "sectors": sectors or [],
               "base_watchlist_version": watchlist_version(current)}
    if coverage:
        payload.update({key: coverage[key] for key in (
            "ranking_status", "ranking_count", "ranking_expected_count",
            "ranking_missing_tickers", "ranking_warnings") if key in coverage})
    proposals_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return "proposal"


def tier_candidates(holdings: list[str], ranked: list[dict], n: int) -> list[str]:
    held = {t.upper() for t in holdings}
    out = []
    for row in ranked:
        tk = row.get("ticker")
        if not tk or tk in held or row.get("error"):
            continue
        out.append(tk)
        if len(out) >= n:
            break
    return out


def _fmt(value, digits=1, suffix=""):
    if value is None:
        return ""
    return f"{value:.{digits}f}{suffix}"


def render_screen_md(as_of: str, ranked: list[dict]) -> str:
    lines = [f"# Daily quant screen — {as_of}", "",
             "Free screen: relative strength vs SPY, distance from the 20/50/200-day average, 14-day ATR, and earnings proximity. No model call.",
             "",
             "| Rank | Ticker | Score | RS 1w | RS 1m | RS 3m | MA20 | MA50 | MA200 | ATR % | Earnings days |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in ranked:
        if row.get("error"):
            lines.append(f"| {row['rank']} | {row['ticker']} |  |  |  |  |  |  |  |  | {row['error']} |")
            continue
        lines.append(
            f"| {row['rank']} | {row['ticker']} | {row.get('score')} | {_fmt(row.get('rs_1w'))} | {_fmt(row.get('rs_1m'))} | "
            f"{_fmt(row.get('rs_3m'))} | {_fmt(row.get('dist_ma20'))} | {_fmt(row.get('dist_ma50'))} | {_fmt(row.get('dist_ma200'))} | "
            f"{_fmt(row.get('atr_pct'))} | {'' if row.get('earnings_days') is None else row['earnings_days']}"
            f"{' live' if row.get('earnings_timing') == 'live' else ''} |")
    return "\n".join(lines) + "\n"


def render_report_md(payload: dict) -> str:
    lines = [f"# Theme scout — {payload['date']}", ""]
    if payload.get("llm"):
        lines.append(f"One deep-model call: `{payload['llm'].get('model')}`.")
    else:
        lines.append("No model call. The watchlist was not changed.")
    lines.append("")
    if payload.get("headline"):
        lines.append(f"**{payload['headline']}**")
        lines.append("")
    lines.append("## Macro")
    macro = payload.get("macro") or {}
    if macro.get("error"):
        lines.append(f"FRED unavailable: {macro['error']}")
    else:
        for key, value in macro.items():
            if not isinstance(value, dict):
                continue
            if value.get("error"):
                lines.append(f"- {key}: {value['error']}")
            else:
                extra = f", YoY {value['yoy_pct']}%" if value.get("yoy_pct") is not None else ""
                flag = " STALE" if value.get("stale") else ""
                lines.append(
                    f"- {key}: {value.get('value')} (obs {value.get('date')}, as_of {value.get('as_of')}, "
                    f"vintage {value.get('vintage')}, age {value.get('age_days')}d{extra}{flag})")
    lines.append("\n## Headlines")
    for item in payload.get("headlines") or []:
        if item.get("error") and not item.get("title"):
            lines.append(f"- {item.get('source')}: {item['error']}")
        elif item.get("title"):
            lines.append(f"- [{item.get('source')}] {item['title']}")
    lines.append("\n## Relative strength vs SPY (percentage points)")
    lines.append("| Group | RS 1w | RS 1m | RS 3m |\n|---|---|---|---|")
    for row in payload.get("relative_strength") or []:
        lines.append(f"| {row['id']} | {_fmt(row.get('rs_1w'))} | {_fmt(row.get('rs_1m'))} | {_fmt(row.get('rs_3m'))} |")
    lines.append("\n## Earnings within 14 days")
    soon = [r for r in payload.get("screen") or [] if isinstance(r.get("earnings_days"), int) and r["earnings_days"] <= 14]
    if soon:
        lines.append(", ".join(f"{r['ticker']} in {r['earnings_days']}d (live calendar, no vintage)" for r in soon))
    else:
        lines.append("None in the screened names.")
    lines.append("\n## Book and watchlist ranking")
    ranking = payload.get("ranking") or []
    if payload.get("ranking_status") == "partial":
        lines.append(f"Partial coverage: {payload.get('ranking_count', len(ranking))} of "
                     f"{payload.get('ranking_expected_count', '?')} expected names returned. "
                     "Missing names have no inferred rank or action.")
    elif payload.get("ranking_status") == "complete":
        lines.append(f"Coverage: all {payload.get('ranking_expected_count', len(ranking))} expected names returned.")
    for warning in payload.get("ranking_warnings") or []:
        lines.append(f"- Ranking warning: {warning}")
    if ranking:
        lines.append("Rank 1 is the strongest returned name. A holding marked trim is a weak position.")
        lines.append("| Rank | Ticker | Bucket | Action | Reason |\n|---|---|---|---|---|")
        ordered = sorted(ranking, key=lambda row: (row.get("rank") if isinstance(row.get("rank"), int) else 10**6, row.get("ticker") or ""))
        for row in ordered:
            reason = str(row.get("reason") or "").replace("|", "/")
            lines.append(f"| {row.get('rank')} | {row.get('ticker')} | {row.get('bucket')} | {row.get('action')} | {reason} |")
    else:
        lines.append("No model ranking.")
    lines.append("\n## Theme stances")
    if payload.get("themes"):
        for theme in payload["themes"]:
            lines.append(f"- **{theme.get('id')}** — {theme.get('stance')}: {theme.get('reason')}")
    else:
        lines.append("Not ranked. This run did not call the model.")
    lines.append("\n## Watchlist changes")
    for warning in payload.get("drop_recommendation_warnings") or []:
        lines.append(f"- Warning: {warning}")
    for warning in payload.get("add_recommendation_mismatches") or []:
        lines.append(f"- Add warning: {warning}")
    changes = payload.get("changes") or []
    if not changes:
        lines.append("None.")
    else:
        for change in changes:
            lines.append(f"- {change.get('action')} {change.get('ticker')}: {change.get('reason')}")
    lines.append(f"\nWatchlist size after this run: {payload.get('watchlist_size')} (band {MIN_WATCH}–{MAX_WATCH}).")
    lines.append("\n## Screen (top 15)")
    lines.append("| Rank | Ticker | Score | RS 1m | Dist MA50 |\n|---|---|---|---|---|")
    for row in (payload.get("screen") or [])[:15]:
        lines.append(f"| {row.get('rank')} | {row.get('ticker')} | {row.get('score')} | {_fmt(row.get('rs_1m'))} | {_fmt(row.get('dist_ma50'))} |")
    return "\n".join(lines) + "\n"


SCOUT_SYSTEM = """You are the theme scout for a concentrated US equity book. You see macro prints, a week of headlines,
relative strength of sectors and AI sub-themes versus SPY, upcoming earnings, and a quant screen of every
holding and every watchlist name.
Rank the WHOLE list in `names`. Rank 1 is the strongest name for this book right now. A strong holding ranks
ahead of a weak one. Put every shown ticker in `ranking` exactly once, with ranks 1 through N and no gaps.
Mark a holding you would keep `keep` and a weak holding you would sell down `trim`. For a candidate, use
`add` when it is worth monitoring on the watchlist and `pass` when it is not worth tracking. An `add` means
research priority only; it is not a buy recommendation, allocation, or order. A candidate can be worth
watching even when its current trend is weak, if a clear event or condition would make it actionable.
Return the structured result promptly. Do not include hidden reasoning or long narrative. Keep each ranking
reason to at most 12 words and each theme reason to at most 20 words; cite the strongest evidence or condition.
For every non-holding candidate marked `add` in `ranking`, include that ticker once in the top-level `add`
array with its sector and a reason it merits monitoring. Do not include `pass` candidates in that array.
Do not answer with new-name ideas only.
Also mark each listed theme overweight, neutral, or avoid. You may propose watchlist adds and drops; do not
force the list to a particular size or invent additions to meet a floor. Only use tickers that appear in
`names` or `screen`. Do not invent a ticker, a listing, or a symbol that was not shown. Proposed adds must
be tickers from that universe. Proposed drops must already be on the watchlist. Never drop a holding; mark
it trim. Explain why a name is worth monitoring and what would change your view. Prefer a focused list of
useful candidates over adding lookalikes. Watchlist proposals require separate user review and application.
This is research support, not an order."""


def agent_plan_from_config() -> dict:
    """The theme-scout seat the next run reads from addons/agent_config.json."""
    root = Path(__file__).resolve().parents[1]
    if not (root / "agent_config.py").is_file():
        root = Path("/home/appuser/addons")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import agent_config
    seat = agent_config.resolved_seat("theme_scout")
    return {"theme_scout": {
        "model": seat.get("model") or DEEP_MODEL,
        "system": seat.get("system") or SCOUT_SYSTEM,
        "max_tokens": seat.get("max_tokens") or 6000,
        "effort": seat.get("effort"),
        "temperature": seat.get("temperature"),
    }}


def call_scout(model: str, context: str, system: str | None = None, max_tokens: int | None = None,
               effort: str | None = None, temperature: float | None = None,
               max_retries: int = 2) -> dict:
    from openai import OpenAI

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "headline": {"type": "string"},
            "themes": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
                "id": {"type": "string"},
                "stance": {"type": "string", "enum": ["overweight", "neutral", "avoid"]},
                "reason": {"type": "string"},
            }, "required": ["id", "stance", "reason"]}},
            "add": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
                "ticker": {"type": "string"}, "sub_theme": {"type": "string"},
                "sector": {"type": "string"}, "reason": {"type": "string"},
            }, "required": ["ticker", "sub_theme", "sector", "reason"]}},
            "drop": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
                "ticker": {"type": "string"}, "reason": {"type": "string"},
            }, "required": ["ticker", "reason"]}},
            "note": {"type": "string"},
            "ranking": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
                "ticker": {"type": "string"},
                "rank": {"type": "integer"},
                "bucket": {"type": "string", "enum": ["holding", "candidate"]},
                "action": {"type": "string", "enum": ["keep", "trim", "add", "pass"]},
                "reason": {"type": "string"},
            }, "required": ["ticker", "rank", "bucket", "action", "reason"]}},
        },
        "required": ["headline", "themes", "add", "drop", "note", "ranking"],
    }
    client = OpenAI(base_url=OPENROUTER_BASE, api_key=key, timeout=900, max_retries=max_retries)
    t0 = time.time()
    extra = {"usage": {"include": True}}
    if effort:
        extra["reasoning"] = {"effort": effort}
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system or SCOUT_SYSTEM},
                  {"role": "user", "content": context + "\n\nReturn the JSON object only."}],
        max_tokens=max_tokens or 6000,
        response_format={"type": "json_schema", "json_schema": {"name": "theme_scout", "strict": True, "schema": schema}},
        extra_body=extra,
    )
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    message = choice.message
    content = message.content or "{}"
    finish_reason = getattr(choice, "finish_reason", None)
    refusal = getattr(message, "refusal", None)
    usage = resp.usage.model_dump() if resp.usage else {}
    try:
        _addons()
        import cost_ledger
        root = Path(os.getenv("FM_DATA_ROOT", str(DATA_ROOT)))
        cost_ledger.write_chat_cost(root, label="theme_scout", model=model, usage=usage)
    except Exception as exc:  # noqa: BLE001
        log("cost file skipped: " + _safe(exc))
    parse_error = None
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        parsed = {}
        parse_error = f"response was not valid JSON: {exc}"
    return {"parsed": parsed, "model": model, "seconds": round(time.time() - t0, 1),
            "usage": usage, "finish_reason": finish_reason, "refusal": refusal,
            "parse_error": parse_error}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def archive_previous_proposal(proposal_path: Path) -> Path | None:
    """Keep the prior proposal for history but remove it from the active review path."""
    if not proposal_path.is_file():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived = proposal_path.with_name(f"watchlist_proposals.superseded_{stamp}.json")
    os.replace(proposal_path, archived)
    return archived


def normalize_ranking(rows, holdings: list[str], candidates: list[str]) -> tuple[list[dict], list[str]]:
    """Validate returned ranking rows without inventing missing ranks.

    Rank values are model output, so gaps and ties are retained. Duplicate tickers
    keep their first valid row. Diagnostics describe partial coverage; callers
    decide whether the usable ranking is empty (failure) or partial (warning).
    """
    errors: list[str] = []
    held: list[str] = []
    for ticker in holdings:
        tk = str(ticker).strip().upper()
        if tk and tk not in held:
            held.append(tk)
    held_set = set(held)
    cands: list[str] = []
    for ticker in candidates:
        tk = str(ticker).strip().upper()
        if tk and tk not in held_set and tk not in cands:
            cands.append(tk)
    expected = held + cands
    by: dict[str, dict] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            errors.append("ranking row is not an object")
            continue
        tk = str(row.get("ticker") or "").strip().upper()
        if not tk:
            errors.append("ranking row missing ticker")
            continue
        if tk in by:
            errors.append(f"duplicate rank row {tk}")
            existing = by[tk]
            duplicate_action = str(row.get("action") or "").strip().lower()
            if duplicate_action != existing["action"]:
                existing["action"] = "conflict"
                existing["action_conflict"] = True
                errors.append(f"conflicting actions for {tk}; marked non-actionable")
            continue
        raw_rank = row.get("rank")
        try:
            rank = int(raw_rank)
            if isinstance(raw_rank, bool) or (isinstance(raw_rank, float) and raw_rank != rank) \
                    or (isinstance(raw_rank, str) and str(rank) != raw_rank.strip()):
                raise ValueError("not an integer")
        except (TypeError, ValueError):
            errors.append(f"{tk} rank is not an integer")
            continue
        if rank < 1:
            errors.append(f"{tk} rank must be a positive integer")
            continue
        action = str(row.get("action") or "").strip().lower()
        if action not in {"keep", "trim", "add", "pass"}:
            errors.append(f"{tk} action {action!r} is not keep, trim, add, or pass")
            continue
        if tk in held_set:
            bucket = "holding"
        elif tk in cands:
            bucket = "candidate"
        else:
            errors.append(f"{tk} was not in the book or the watchlist")
            continue
        by[tk] = {
            "ticker": tk, "rank": rank, "bucket": bucket, "action": action,
            "reason": str(row.get("reason") or "").strip(),
        }
    missing = [tk for tk in expected if tk not in by]
    if missing:
        errors.append("missing from ranking: " + ", ".join(missing))
    if expected and len(by) < len(expected):
        errors.append(f"partial ranking: {len(by)} of {len(expected)} expected names returned")
    ordered = sorted(by.values(), key=lambda row: row["rank"])
    return ordered, errors


def model_screen_evidence(ranked: list[dict]) -> list[dict]:
    """Keep only fields needed to rank names; the full screen remains in the saved report."""
    fields = ("ticker", "score", "rs_1w", "rs_1m", "rs_3m", "dist_ma20", "dist_ma50",
              "dist_ma200", "atr_pct", "earnings_days", "error")
    return [{key: row.get(key) for key in fields if key in row} for row in ranked]


def compact_scout_headlines(headlines: list[dict], per_source: int = 4) -> tuple[list[dict], dict]:
    """Give each configured news source room and preserve its error status."""
    expected_sources = ("yfinance", "alpha_vantage")
    grouped: dict[str, list[dict]] = {name: [] for name in expected_sources}
    for row in headlines:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source") or "unknown")
        grouped.setdefault(source, []).append(row)

    compact = []
    statuses = {}
    for source, rows in grouped.items():
        errors = [str(row.get("error") or "").strip() for row in rows if row.get("error")]
        titles = [row for row in rows if row.get("title") and not row.get("error")]
        if errors:
            status = {"status": "error", "message": (errors[0] or "source returned an error without details")[:180]}
        elif titles:
            status = {"status": "ok", "headlines_available": len(titles)}
        else:
            status = {"status": "no_headlines", "headlines_available": 0}
        statuses[source] = status

        if errors:
            compact.append({"source": source, "error": status["message"]})
        for row in titles[:per_source]:
            item = {"source": source, "title": str(row.get("title") or "").strip()[:200]}
            for key in ("ticker", "date", "headline_date", "timing"):
                if row.get(key) is not None:
                    item[key] = row[key]
            if row.get("summary"):
                item["summary"] = str(row["summary"]).strip()[:180]
            compact.append(item)
    return compact, statuses


def scout_model_context(date_text: str, macro: dict, headlines: list[dict], strength: list[dict],
                       sectors: list[str], names: list[dict], ranked: list[dict],
                       subthemes: list[str]) -> str:
    """Compact duplicate screen data while retaining every holding and candidate."""
    evidence = {row.get("ticker"): row for row in model_screen_evidence(ranked)}
    model_names = [{**name, "evidence": evidence.get(name["ticker"], {})} for name in names]
    headline_rows, headline_sources = compact_scout_headlines(headlines)
    return json.dumps({
        "date": date_text,
        "macro": macro,
        "headlines": headline_rows,
        "headline_sources": headline_sources,
        "relative_strength": strength,
        "discovery_sectors": sectors,
        "names": model_names,
        "subthemes": subthemes,
        "ranking_count": len(model_names),
        "instructions": "Rank every name exactly once. Keep reasons concise. Missing evidence is unknown, not zero.",
    }, separators=(",", ":"), ensure_ascii=False, default=str)


def ranking_response_error(result: dict) -> str | None:
    """Reject truncated or empty model output before it can become a proposal."""
    if result.get("finish_reason") == "length":
        return "model response reached its output-token limit; ranking is incomplete"
    if result.get("refusal"):
        return "model refused the request: " + str(result["refusal"])
    if result.get("parse_error"):
        return "model response could not be parsed: " + str(result["parse_error"])
    parsed = result.get("parsed")
    if not isinstance(parsed, dict) or not isinstance(parsed.get("ranking"), list) or not parsed["ranking"]:
        return "model response contained no ranking"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Theme scout and daily quant screen")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--themes", default="")
    ap.add_argument("--screen", action="store_true", help="daily quant screen only")
    ap.add_argument("--no-llm", action="store_true", help="weekly pack without a model call or watchlist edit")
    ap.add_argument("--dry-run", action="store_true", help="same as --no-llm")
    ap.add_argument("--tier-n", type=int, default=8)
    ap.add_argument("--sectors", nargs="*", default=[], choices=SECTORS)
    ap.add_argument("--model", default=None, help="omit to use addons/agent_config.json")
    ap.add_argument("--keep-watchlist", action="store_true",
                    help="save the scout report and do not rewrite watchlist.json")
    ap.add_argument("--single-call", action="store_true",
                    help="disable client retries for one explicitly scoped paid scout request")
    args = ap.parse_args()
    args.sectors = list(dict.fromkeys(args.sectors or SECTORS))
    if not args.sectors or len(args.sectors) > len(SECTORS):
        ap.error(f"choose between one and {len(SECTORS)} discovery sectors")
    scout = agent_plan_from_config()["theme_scout"]
    scout_model = args.model or scout["model"]
    no_llm = args.no_llm or args.dry_run or args.screen

    root = Path(args.data_root)
    book = load_json(root / "my_book.json", {"positions": []}) or {"positions": []}
    holdings = [str(p["ticker"]).upper() for p in book.get("positions") or []]
    watch_path = root / "watchlist.json"
    watch = load_watchlist(watch_path)
    themes_file = _themes_path(args.themes)
    themes = load_json(themes_file, {}) if themes_file else {}
    subthemes = (themes or {}).get("subthemes") or {}

    discovered = discover_sector_candidates(args.sectors)
    discovered_by = {row["ticker"]: row for row in discovered}
    discovered = [row for row in discovered if row["ticker"] not in set(holdings) | {w["ticker"] for w in watch}]
    universe = list(dict.fromkeys(holdings + [w["ticker"] for w in watch]
                                  + [row["ticker"] for row in discovered] + args.sectors + ["SPY"]))
    log(f"downloading prices for {len(universe)} symbols")
    ohlc = download_ohlc(universe, args.date)
    close, high, low = ohlc["Close"], ohlc["High"], ohlc["Low"]
    screen_names = list(dict.fromkeys(holdings + [w["ticker"] for w in watch]
                                      + [row["ticker"] for row in discovered] + args.sectors))
    etf_names = {w["ticker"] for w in watch if (w.get("type") or "").lower() == "etf"} | set(args.sectors) | {"SPY"}
    earnings = {}
    for tk in screen_names:
        if tk in etf_names:
            continue
        earnings[tk] = earnings_days(tk, datetime.strptime(args.date, "%Y-%m-%d").date())
    ranked = screen_frame(close, high, low, screen_names, "SPY", earnings)
    screened_tickers = {str(row.get("ticker") or "").upper() for row in ranked
                        if isinstance(row.get("score"), (int, float)) and math.isfinite(float(row["score"]))
                        and not row.get("error")}
    screened_discovered = [row for row in discovered if row["ticker"] in screened_tickers]
    screened_sector_etfs = [{"ticker": ticker, "sector": SECTOR_NAMES[ticker], "type": "etf",
                             "discovery_source": "selected sector ETF"}
                            for ticker in args.sectors if ticker in screened_tickers]
    groups = {"sector_" + s: [s] for s in args.sectors}
    groups.update(subthemes)
    strength = group_rs(close, groups, "SPY")
    tier = tier_candidates(holdings, ranked, args.tier_n)

    out_dir = root / "theme_scout" / args.date
    screen_payload = {"date": args.date, "ranked": ranked, "tier_candidates": tier, "tier_n": args.tier_n}
    _write(out_dir / "screen.json", json.dumps(screen_payload, indent=2))
    _write(out_dir / "screen.md", render_screen_md(args.date, ranked))
    print("SCREEN_PATH=" + str(out_dir / "screen.json"))
    print("TIER_CANDIDATES=" + ",".join(tier))

    if args.screen and not args.no_llm and not args.dry_run:
        log(f"screen only: {out_dir / 'screen.md'}")
        return 0

    _addons()
    import macro_panel
    macro = fred_snapshot(args.date)
    headlines = macro_panel.filter_headlines(yahoo_headlines(["SPY", "QQQ", "SMH"]) + av_headlines(args.date), args.date)
    for item in headlines:
        if item.get("source") == "yfinance":
            item["timing"] = "live"
    payload = {
        "date": args.date,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_skipped": no_llm,
        "macro": macro,
        "headlines": headlines,
        "relative_strength": strength,
        "screen": ranked,
        "discovery_pool": screened_discovered,
        "tier_candidates": tier,
        "themes": [],
        "changes": [],
        "headline": "",
        "watchlist_size": len(watch),
        "sectors": args.sectors,
        "llm": None,
    }
    if not no_llm:
        held_set = set(holdings)
        names = [{"ticker": tk, "bucket": "holding"} for tk in holdings]
        names += [{"ticker": w["ticker"], "bucket": "candidate", "sector": w.get("sector"),
                   "type": w.get("type"), "sub_theme": w.get("sub_theme")}
                  for w in watch if w["ticker"] not in held_set]
        names += [{"ticker": row["ticker"], "bucket": "discovery", "sector": row["sector"],
                   "type": row["type"], "discovery_source": row["discovery_source"]}
                  for row in screened_discovered]
        names += [{"ticker": row["ticker"], "bucket": "candidate", "sector": row["sector"],
                   "type": row["type"], "discovery_source": row["discovery_source"]}
                  for row in screened_sector_etfs if row["ticker"] not in held_set
                  and row["ticker"] not in {item["ticker"] for item in watch}]
        context = scout_model_context(args.date, macro, headlines, strength, args.sectors,
                                      names, ranked, list(subthemes))
        rank_tokens = max(int(scout["max_tokens"] or 6000), 1)
        log(f"one scout call ({scout_model})")
        retry_limit = 0 if args.single_call else 2
        if args.single_call:
            log("single-call mode: automatic client retries disabled")
        log(f"agent config: theme_scout={scout_model} effort={scout['effort']} max_tokens={rank_tokens}")
        log(f"ranking {len(names)} names ({len(holdings)} holdings + {len(names) - len(holdings)} candidates)")
        superseded = archive_previous_proposal(out_dir / "watchlist_proposals.json")
        if superseded:
            payload["superseded_proposal"] = superseded.name
            log("previous same-day proposal archived before the new model request")
        try:
            result = call_scout(scout_model, context, scout["system"], rank_tokens, scout["effort"],
                                scout["temperature"], max_retries=retry_limit)
        except Exception as exc:  # noqa: BLE001
            log("scout call failed: " + _safe(exc))
            return 1
        parsed = result["parsed"]
        payload["llm"] = {"model": scout_model, "seconds": result["seconds"], "usage": result["usage"],
                           "finish_reason": result.get("finish_reason")}
        payload["headline"] = parsed.get("headline") or ""
        payload["themes"] = parsed.get("themes") or []
        ranking_candidates = ([w["ticker"] for w in watch] + [row["ticker"] for row in screened_discovered]
                              + [row["ticker"] for row in screened_sector_etfs])
        response_error = ranking_response_error(result)
        if response_error:
            ordered, rank_errors = [], [response_error]
        else:
            ordered, rank_errors = normalize_ranking(parsed["ranking"], holdings, ranking_candidates)
            if not ordered:
                response_error = "model response contained no valid ranking rows"
        expected_tickers = list(dict.fromkeys(
            str(t).strip().upper() for t in holdings + ranking_candidates if str(t).strip()))
        returned_tickers = {row["ticker"] for row in ordered}
        missing_tickers = [ticker for ticker in expected_tickers if ticker not in returned_tickers]
        payload["ranking_status"] = ("failed" if response_error else
                                     "partial" if missing_tickers or rank_errors else "complete")
        payload["ranking_count"] = len(ordered)
        payload["ranking_expected_count"] = len(expected_tickers)
        payload["ranking_missing_tickers"] = missing_tickers
        payload["ranking_warnings"] = rank_errors
        payload["ranking"] = ordered or (parsed.get("ranking") or [])
        payload["ranking_errors"] = rank_errors
        usage = result.get("usage") or {}
        log(f"scout done in {result['seconds']}s tokens_in={usage.get('prompt_tokens')} tokens_out={usage.get('completion_tokens')} cost={usage.get('cost')}")
        if response_error:
            payload["model_response_error"] = response_error
            for err in rank_errors:
                log("ranking rejected: " + err)
            _write(out_dir / "report.json", json.dumps(payload, indent=2, default=str))
            _write(out_dir / "report.md", render_report_md(payload))
            print("REPORT_PATH=" + str(out_dir / "report.md"))
            return 1
        scores = {r["ticker"]: r.get("score") if r.get("score") is not None else -1e9 for r in ranked}
        if args.keep_watchlist:
            payload["changes"] = []
            payload["watchlist_size"] = len(watch)
            log("watchlist left unchanged")
        else:
            _addons()
            import watchlist_log
            targets = load_json(root / "targets.json", {}) or {}
            staged = staged_tickers(root)
            universe = set(holdings) | {w["ticker"] for w in watch} | set(discovered_by) | set(SECTORS) | {"SPY"}
            additions_by_ticker = {row["ticker"]: row for row in screened_discovered + screened_sector_etfs
                                   if row["ticker"] not in set(holdings)}
            additions_by_ticker.update({row["ticker"]: row for row in watch})
            additions, add_mismatches = ranked_watchlist_additions(
                ordered, parsed.get("add") or [], additions_by_ticker, set(holdings))
            if add_mismatches:
                payload["add_recommendation_mismatches"] = add_mismatches
                for mismatch in add_mismatches:
                    log("watchlist add mismatch: " + mismatch)
            proposed_drops = parsed.get("drop") or []
            safe_drops, drop_warnings = ranked_watchlist_drops(ordered, proposed_drops)
            if drop_warnings:
                payload["drop_recommendation_warnings"] = drop_warnings
            updated, changes = apply_watchlist(
                watch, additions, safe_drops, scores,
                as_of=args.date, holdings=set(holdings), staged=staged,
                universe=universe, log_events=watchlist_log.read_events(root),
                targets=targets, themes=themes,
                ai_lookthrough_pct=book_ai_lookthrough(book, close, targets, themes))
            payload["changes"] = changes
            payload["watchlist_size"] = len(watch)
            proposals = out_dir / "watchlist_proposals.json"
            outcome = commit_watchlist(
                watch_path, watch, updated, changes, proposals, root, args.date, args.sectors,
                coverage={key: payload[key] for key in (
                    "ranking_status", "ranking_count", "ranking_expected_count",
                    "ranking_missing_tickers", "ranking_warnings")})
            if outcome == "proposal":
                log(f"proposal at {proposals.name}; watchlist not edited")
        top = [row["ticker"] for row in ordered
               if row["bucket"] == "candidate" and not row.get("action_conflict")][: args.tier_n]
        payload["top_non_holdings"] = top
        print("TOP_NON_HOLDINGS=" + ",".join(top))
        print("HOLDING_RANKS=" + ",".join(
            f"{row['ticker']}:{row['rank']}:{row['action']}" for row in ordered if row["bucket"] == "holding"))
    _write(out_dir / "report.json", json.dumps(payload, indent=2, default=str))
    _write(out_dir / "report.md", render_report_md(payload))
    if payload.get("ranking") and payload.get("ranking_status") in {"complete", "partial"}:
        lines = [f"# Theme scout context {payload['date']}", "", str(payload.get("headline") or ""), ""]
        if payload.get("ranking_status") == "partial":
            lines += [f"Partial ranking: {payload.get('ranking_count')} of "
                      f"{payload.get('ranking_expected_count')} expected names. Missing names have no inferred rank or action.", ""]
        for theme in payload.get("themes") or []:
            lines.append(f"- {theme.get('id')}: {theme.get('stance')} — {theme.get('reason')}")
        lines.append("")
        lines.append("| Rank | Ticker | Bucket | Action | Reason |")
        lines.append("|---|---|---|---|---|")
        for row in payload["ranking"]:
            reason = str(row.get("reason") or "").replace("|", "/")
            if len(reason) > 160:
                reason = reason[:157] + "..."
            lines.append(f"| {row['rank']} | {row['ticker']} | {row['bucket']} | {row['action']} | {reason} |")
        _write(out_dir / "context.md", "\n".join(lines) + "\n")
        print("CONTEXT_PATH=" + str(out_dir / "context.md"))
    log(f"report: {out_dir / 'report.md'}")
    print("REPORT_PATH=" + str(out_dir / "report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
