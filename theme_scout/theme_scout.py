#!/usr/bin/env python3
"""Weekly theme scout and daily quant screen. Does not modify TradingAgents source.

--screen   free ranked screen only
--no-llm   weekly pack (macro, headlines, earnings, relative strength, screen) and no model call
--dry-run  same as --no-llm: no model call and no watchlist write

A weekly run without those flags makes one deep-model call and then updates the watchlist.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(os.getenv("FM_DATA_ROOT", "/home/appuser/.tradingagents"))
DEEP_MODEL = os.getenv("THEME_SCOUT_MODEL", "openai/gpt-6.1-sol")
OPENROUTER_BASE = os.getenv("FUND_MANAGER_BASE_URL", "https://openrouter.ai/api/v1")
MIN_WATCH = 30
MAX_WATCH = 50
SECTORS = ["XLK", "XLY", "XLV", "XLP", "XLU", "XLE", "XLF", "SMH"]
WINDOWS = {"1w": 5, "1m": 21, "3m": 63}
HISTORY_DAYS = 430  # about 14 months of calendar days


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
        raw = raw.get("candidates") or raw.get("watchlist") or raw.get("tickers") or []
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


def apply_watchlist(items: list[dict], adds: list[dict], drops: list[dict],
                    scores: dict[str, float], min_n: int = MIN_WATCH, max_n: int = MAX_WATCH) -> tuple[list[dict], list[dict]]:
    """Apply proposed adds and drops, then keep the list inside min_n..max_n."""
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
    for add in adds or []:
        tk = str(add.get("ticker") or "").strip().upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", tk or ""):
            changes.append({"action": "skip_add", "ticker": tk, "reason": "invalid ticker"})
            continue
        if tk in by:
            changes.append({"action": "skip_add", "ticker": tk, "reason": "already on the watchlist"})
            continue
        row = {"ticker": tk, "sector": add.get("sector") or "Unclassified", "type": add.get("type") or "stock"}
        if add.get("sub_theme"):
            row["sub_theme"] = add["sub_theme"]
        if add.get("reason"):
            row["note"] = add["reason"]
        by[tk] = row
        order.append(tk)
        fresh.add(tk)
        changes.append({"action": "add", "ticker": tk, "reason": add.get("reason") or ""})
    for drop in drops or []:
        tk = str(drop.get("ticker") if isinstance(drop, dict) else drop or "").strip().upper()
        reason = drop.get("reason") if isinstance(drop, dict) else ""
        if tk not in by:
            changes.append({"action": "skip_drop", "ticker": tk, "reason": "not on the watchlist"})
            continue
        if len(by) - 1 < min_n:
            changes.append({"action": "kept", "ticker": tk, "reason": f"dropping would put the list under {min_n}"})
            continue
        by.pop(tk)
        order = [t for t in order if t != tk]
        changes.append({"action": "drop", "ticker": tk, "reason": reason})
    while len(by) > max_n:
        pool = [tk for tk in by if tk not in fresh] or list(by)
        victim = min(pool, key=lambda tk: scores.get(tk, -1e9))
        by.pop(victim)
        fresh.discard(victim)
        order = [t for t in order if t != victim]
        changes.append({"action": "trim", "ticker": victim, "reason": f"watchlist cap is {max_n}"})
    if len(by) < min_n:
        changes.append({"action": "short", "ticker": "", "reason": f"{len(by)} names is under the {min_n} floor. Nothing was invented to fill it."})
    return [by[tk] for tk in order if tk in by], changes


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
    if ranking:
        lines.append("Rank 1 is the strongest name. A holding marked trim is a weak position.")
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
Mark a holding you would keep `keep` and a weak holding you would sell down `trim`. Mark a candidate you
would buy `add` and one you would not `pass`. Do not answer with new-name ideas only.
Also mark each listed theme overweight, neutral, or avoid. You may propose watchlist adds and drops so the
list stays between 30 and 50 names. Do not invent tickers you were not shown unless they are liquid US
listings you name explicitly. Prefer dropping weak relative-strength names over adding lookalikes. This is
research support, not an order."""


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
               effort: str | None = None, temperature: float | None = None) -> dict:
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
    client = OpenAI(base_url=OPENROUTER_BASE, api_key=key, timeout=900, max_retries=2)
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
    content = resp.choices[0].message.content or "{}"
    usage = resp.usage.model_dump() if resp.usage else {}
    return {"parsed": json.loads(content), "model": model, "seconds": round(time.time() - t0, 1), "usage": usage}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_ranking(rows, holdings: list[str], candidates: list[str]) -> tuple[list[dict], list[str]]:
    """Every holding and candidate once, ranks 1..N. Rank 1 is the strongest name."""
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
            continue
        try:
            rank = int(row.get("rank"))
        except (TypeError, ValueError):
            errors.append(f"{tk} rank is not an integer")
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
    ranks = [by[tk]["rank"] for tk in by]
    if len(ranks) != len(set(ranks)):
        errors.append("ranks are not unique")
    if expected and (len(by) != len(expected) or not ranks or min(ranks) != 1 or max(ranks) != len(expected)):
        errors.append(f"ranks must be 1 through {len(expected)}")
    ordered = sorted(by.values(), key=lambda row: row["rank"])
    return ordered, errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Theme scout and daily quant screen")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--themes", default="")
    ap.add_argument("--screen", action="store_true", help="daily quant screen only")
    ap.add_argument("--no-llm", action="store_true", help="weekly pack without a model call or watchlist edit")
    ap.add_argument("--dry-run", action="store_true", help="same as --no-llm")
    ap.add_argument("--tier-n", type=int, default=8)
    ap.add_argument("--model", default=None, help="omit to use addons/agent_config.json")
    ap.add_argument("--keep-watchlist", action="store_true",
                    help="save the scout report and do not rewrite watchlist.json")
    args = ap.parse_args()
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

    universe = list(dict.fromkeys(holdings + [w["ticker"] for w in watch] + SECTORS + ["SPY"]))
    log(f"downloading prices for {len(universe)} symbols")
    ohlc = download_ohlc(universe, args.date)
    close, high, low = ohlc["Close"], ohlc["High"], ohlc["Low"]
    screen_names = list(dict.fromkeys(holdings + [w["ticker"] for w in watch]))
    etf_names = {w["ticker"] for w in watch if (w.get("type") or "").lower() == "etf"} | set(SECTORS) | {"SPY"}
    earnings = {}
    for tk in screen_names:
        if tk in etf_names:
            continue
        earnings[tk] = earnings_days(tk, datetime.strptime(args.date, "%Y-%m-%d").date())
    ranked = screen_frame(close, high, low, screen_names, "SPY", earnings)
    groups = {"sector_" + s: [s] for s in SECTORS}
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
        "tier_candidates": tier,
        "themes": [],
        "changes": [],
        "headline": "",
        "watchlist_size": len(watch),
        "llm": None,
    }
    if not no_llm:
        held_set = set(holdings)
        names = [{"ticker": tk, "bucket": "holding"} for tk in holdings]
        names += [{"ticker": w["ticker"], "bucket": "candidate", "sector": w.get("sector"),
                   "type": w.get("type"), "sub_theme": w.get("sub_theme")}
                  for w in watch if w["ticker"] not in held_set]
        context = json.dumps({
            "date": args.date,
            "macro": macro,
            "headlines": headlines[:24],
            "relative_strength": strength,
            "earnings_within_14d": [r for r in ranked if isinstance(r.get("earnings_days"), int) and r["earnings_days"] <= 14],
            "names": names,
            "screen": ranked,
            "subthemes": list(subthemes),
            "rank_count": len(names),
        }, indent=1, default=str)
        rank_tokens = max(int(scout["max_tokens"] or 6000), 16000)
        log(f"one scout call ({scout_model})")
        log(f"agent config: theme_scout={scout_model} effort={scout['effort']} max_tokens={rank_tokens}")
        log(f"ranking {len(names)} names ({len(holdings)} holdings + {len(names) - len(holdings)} candidates)")
        try:
            result = call_scout(scout_model, context, scout["system"], rank_tokens, scout["effort"], scout["temperature"])
        except Exception as exc:  # noqa: BLE001
            log("scout call failed: " + _safe(exc))
            return 1
        parsed = result["parsed"]
        payload["llm"] = {"model": scout_model, "seconds": result["seconds"], "usage": result["usage"]}
        payload["headline"] = parsed.get("headline") or ""
        payload["themes"] = parsed.get("themes") or []
        ordered, rank_errors = normalize_ranking(parsed.get("ranking") or [], holdings, [w["ticker"] for w in watch])
        payload["ranking"] = ordered or (parsed.get("ranking") or [])
        payload["ranking_errors"] = rank_errors
        usage = result.get("usage") or {}
        log(f"scout done in {result['seconds']}s tokens_in={usage.get('prompt_tokens')} tokens_out={usage.get('completion_tokens')} cost={usage.get('cost')}")
        if rank_errors:
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
            updated, changes = apply_watchlist(watch, parsed.get("add") or [], parsed.get("drop") or [], scores)
            payload["changes"] = changes
            payload["watchlist_size"] = len(updated)
            if changes and watch_path.exists():
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = watch_path.with_name(watch_path.name + f".bak_{stamp}")
                shutil.copy2(watch_path, backup)
                payload["watchlist_backup"] = str(backup)
                watch_path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
                log(f"watchlist updated ({len(watch)} -> {len(updated)}); backup {backup.name}")
            elif changes:
                log("changes were computed but the watchlist file is missing, so nothing was written")
        top = [row["ticker"] for row in ordered if row["bucket"] == "candidate"][: args.tier_n]
        payload["top_non_holdings"] = top
        print("TOP_NON_HOLDINGS=" + ",".join(top))
        print("HOLDING_RANKS=" + ",".join(
            f"{row['ticker']}:{row['rank']}:{row['action']}" for row in ordered if row["bucket"] == "holding"))
    _write(out_dir / "report.json", json.dumps(payload, indent=2, default=str))
    _write(out_dir / "report.md", render_report_md(payload))
    if payload.get("ranking") and not payload.get("ranking_errors"):
        lines = [f"# Theme scout context {payload['date']}", "", str(payload.get("headline") or ""), ""]
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
