#!/usr/bin/env python3
"""Fund Manager add-on for TradingAgents (does not modify the native project).

Pipeline:
  1. latest per-ticker TradingAgents reports (holdings + optional watchlist candidates)
  2. quantitative portfolio picture (prices, weights, P&L, correlations, vol, beta, VaR,
     stress, concentration, drift, market regime, FRED macro, candidate diversification)
  3. Fund Manager (Sol) writes a DRAFT whole-portfolio plan incl. a dynamic cash target
  4. portfolio risk team (Aggressive / Conservative / Neutral) debates the draft, N rounds
  5. Fund Manager reads the debate and issues the FINAL plan (with changes vs draft)

Runs inside the tradingagents Docker image (reuses yfinance/pandas/openai and .env).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ADDON_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("FM_DATA_ROOT", "/home/appuser/.tradingagents"))
DEFAULT_MODEL = os.getenv("FUND_MANAGER_MODEL", "openai/gpt-6.1-sol")
# Quick model already used for per-stock runs (cli prefs: openai/gpt-6-luna).
# The conservative seat used to default to anthropic/claude-opus-5.5 (~$0.39 of a
# ~$1.20 committee). Opus is not a default anywhere in this add-on: it is the
# expensive seat and easy to leave on. Luna takes that seat, and also the neutral
# seat so the neutral voice is not the fund manager's own Sol model.
QUICK_MODEL = "openai/gpt-6-luna"
OPENROUTER_BASE = os.getenv("FUND_MANAGER_BASE_URL", "https://openrouter.ai/api/v1")
BENCH = "SPY"
TRADING_DAYS = 252
REGIME_SYMBOLS = ["QQQ", "SMH", "IWM", "^VIX", "^VIX3M"]
# GOOG/MSFT/AAPL sit in megacap_quality, not ai_growth. Count them, plus this
# fraction of SPY (mega-cap tech share of the index), in the AI sleeve.
AI_MEGACAP = {"AAPL", "MSFT", "GOOG", "GOOGL"}
SPY_TECH_SHARE = 0.33
FRED_LOOKBACK_DAYS = 800
FRED_MONTHLY = {"fed_funds", "core_cpi_index", "unemployment"}
RISK_POLICY = {
    "single_stock_cap_pct": 12.0,
    "high_vol_cap_pct": 3.0,
    "high_vol_threshold_pct": 60.0,
    "short_history_cap_pct": 2.0,
    "short_history_bars": 252,
    "broad_index_cap_pct": 35.0,
    "ai_sleeve_cap_pct": 35.0,
    "spy_tech_share": SPY_TECH_SHARE,
    "ai_megacap": sorted(AI_MEGACAP),
    "top3_cap_pct": 50.0,
    "min_trade_usd": 500.0,
    "earnings_warn_days": 7,
}


def log(msg: str) -> None:
    print(f"[fund-manager] {msg}", flush=True)


# --------------------------------------------------------------------------- inputs
def load_json(path: Path, default=None):
    if path and path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def find_latest_reports(reports_dir: Path, tickers: list[str], since: str | None) -> dict:
    """Latest report folder per ticker: <TICKER>_<YYYYMMDD>_<HHMMSS>."""
    latest: dict[str, tuple[str, Path]] = {}
    if not reports_dir.exists():
        return {}
    for d in reports_dir.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"^(.+)_(\d{8}_\d{6})$", d.name)
        if not m:
            continue
        tk, stamp = m.group(1), m.group(2)
        if tk not in tickers:
            continue
        if since and stamp[:8] < since.replace("-", ""):
            continue
        if not (d / "complete_report.md").exists():
            continue
        if tk not in latest or stamp > latest[tk][0]:
            latest[tk] = (stamp, d)
    return {tk: v[1] for tk, v in latest.items()}


def _read(p: Path, limit: int | None = None) -> str:
    if not p.exists():
        return ""
    txt = p.read_text(encoding="utf-8", errors="replace").strip()
    if limit and len(txt) > limit:
        txt = txt[:limit] + " …[truncated]"
    return txt


def _field(text: str, name: str) -> str | None:
    m = re.search(rf"\*\*{re.escape(name)}\*\*:\s*(.+)", text)
    return m.group(1).strip() if m else None


def _clip(t: str | None, n: int = 220) -> str:
    t = re.sub(r"\s+", " ", (t or "")).strip().replace("|", "/")
    return t if len(t) <= n else t[: n - 1].rsplit(" ", 1)[0] + "…"


def _gist_debater(text: str) -> str:
    """Headline + final recommendation line of a bull/bear debater note."""
    if not text:
        return ""
    head = re.search(r"##\s*(.+)", text)
    rec = re.findall(r"\*\*Recommendation:?\*\*:?\s*(.+)|\*\*Recommendation:\s*(.+?)\*\*", text)
    rec_txt = ""
    if rec:
        last = rec[-1]
        rec_txt = (last[0] or last[1]).strip("* ")
    return _clip((head.group(1) if head else "") + (" | " + rec_txt if rec_txt else ""), 240)


def _gist_last_para(text: str) -> str:
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip() and not p.strip().startswith("#")]
    return _clip(re.sub(r"^\w+ Analyst:\s*", "", paras[-1]) if paras else "", 200)


def parse_report(tk: str, d: Path, include_risk: bool, max_chars: int) -> dict:
    header = _read(d / "complete_report.md")[:1500]
    pm = _read(d / "5_portfolio" / "decision.md")
    trader = _read(d / "3_trading" / "trader.md")
    research = _read(d / "2_research" / "manager.md")
    if not pm:  # fall back to the PM section of the complete report
        full = _read(d / "complete_report.md")
        idx = full.find("## V. Portfolio Manager Decision")
        pm = full[idx:] if idx >= 0 else ""
    analysis_date = re.search(r"Analysis date:\s*(\S+)", header)
    generated = re.search(r"Generated:\s*(.+)", header)
    rating = _field(pm, "Rating") or (re.search(r"- Rating:\s*(.+)", header) or [None, None])[1]
    out = {
        "ticker": tk,
        "report_dir": d.name,
        "analysis_date": analysis_date.group(1) if analysis_date else None,
        "generated_utc": generated.group(1).strip() if generated else None,
        "pm_rating": rating,
        "pm_price_target": _field(pm, "Price Target"),
        "pm_time_horizon": _field(pm, "Time Horizon"),
        "trader_action": _field(trader, "Action"),
        "trader_position_sizing": _field(trader, "Position Sizing"),
        "trader_stop_loss": _field(trader, "Stop Loss"),
        "trader_entry": _field(trader, "Entry Price"),
        "pm_decision": _read(d / "5_portfolio" / "decision.md", max_chars) or pm[:max_chars],
        "research_manager": _read(d / "2_research" / "manager.md", max_chars // 2),
    }
    out["gists"] = {
        "bull": _gist_debater(_read(d / "2_research" / "bull.md")),
        "bear": _gist_debater(_read(d / "2_research" / "bear.md")),
        "research_manager": _field(research, "Recommendation"),
        **{f"risk_{k}": _gist_last_para(_read(d / "4_risk" / f"{k}.md"))
           for k in ("aggressive", "conservative", "neutral")},
    }
    if include_risk:
        out["risk_debate"] = {
            k: _read(d / "4_risk" / f"{k}.md", max_chars // 3)
            for k in ("aggressive", "conservative", "neutral")
        }
    return out


# --------------------------------------------------------------------------- market data
def fetch_prices(tickers: list[str], period: str = "2y") -> pd.DataFrame:
    import yfinance as yf

    syms = sorted(set(tickers + [BENCH]))
    for attempt in range(3):
        try:
            df = yf.download(syms, period=period, interval="1d", auto_adjust=True,
                             progress=False, group_by="column", threads=True)
            close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
            if isinstance(close, pd.Series):
                close = close.to_frame(syms[0])
            close = close.dropna(how="all")
            if not close.empty:
                return close
        except Exception as e:  # noqa: BLE001
            log(f"yfinance download failed (attempt {attempt + 1}): {e}")
        time.sleep(3)
    raise RuntimeError("Could not download price history from yfinance")


def fetch_prices_with_regime(tickers: list[str]) -> pd.DataFrame:
    """Holdings plus the regime tape. A bad index symbol must not sink the book download."""
    try:
        return fetch_prices(list(tickers) + REGIME_SYMBOLS)
    except RuntimeError:
        log("price download failed with regime symbols; retrying holdings only")
        return fetch_prices(list(tickers))


def live_quotes(tickers: list[str], close: pd.DataFrame) -> dict:
    """Best-effort latest price; falls back to last daily close."""
    import yfinance as yf

    out = {}
    for tk in tickers:
        px, src = None, None
        try:
            fi = yf.Ticker(tk).fast_info
            px = float(fi["last_price"]) if fi["last_price"] else None
            src = "yfinance fast_info last_price"
        except Exception:  # noqa: BLE001
            px = None
        if (px is None or not math.isfinite(px)) and tk in close:
            s = close[tk].dropna()
            if len(s):
                px, src = float(s.iloc[-1]), f"last close {s.index[-1].date()}"
        out[tk] = {"price": px, "source": src}
    return out


def ticker_meta(tickers: list[str]) -> dict:
    import yfinance as yf

    meta = {}
    for tk in tickers:
        try:
            i = yf.Ticker(tk).info
            meta[tk] = {"name": i.get("longName") or i.get("shortName"), "type": i.get("quoteType"),
                        "sector": i.get("sector"), "industry": i.get("industry")}
        except Exception:  # noqa: BLE001
            meta[tk] = {"name": None, "type": None, "sector": None, "industry": None}
    return meta


def fred_snapshot(as_of: str) -> dict:
    """Latest values of a few macro series via the project's own FRED vendor."""
    try:
        from tradingagents.dataflows.vendors import fred
    except Exception as e:  # noqa: BLE001
        return {"error": f"fred vendor import failed: {e}"}
    series = {"fed_funds": "FEDFUNDS", "10y_treasury": "DGS10", "2y_treasury": "DGS2",
              "10y_2y_spread": "T10Y2Y", "vix": "VIXCLS", "core_cpi_index": "CPILFESL",
              "unemployment": "UNRATE", "dollar_index": "DTWEXBGS"}
    out = {}
    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")
    # 800 days so a lagged monthly print still has an observation ~365 days earlier (CPI YoY).
    start = (as_of_dt - timedelta(days=FRED_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    for name, sid in series.items():
        try:
            data = fred._request("series/observations", {
                "series_id": sid, "observation_start": start, "observation_end": as_of,
                "sort_order": "desc", "limit": 900})
            obs = [o for o in data.get("observations", []) if o.get("value") not in (".", None)]
            if not obs:
                continue
            latest = obs[0]
            age = (as_of_dt - datetime.strptime(latest["date"], "%Y-%m-%d")).days
            limit = 45 if name in FRED_MONTHLY else 4
            entry = {"date": latest["date"], "value": float(latest["value"]), "age_days": age,
                     "stale": age > limit}
            # value ~1 year earlier for context
            target = (datetime.strptime(latest["date"], "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
            older = [o for o in obs if o["date"] <= target]
            if older:
                entry["value_1y_ago"] = float(older[0]["value"])
                if sid == "CPILFESL":
                    entry["yoy_pct"] = round((entry["value"] / entry["value_1y_ago"] - 1) * 100, 2)
            out[name] = entry
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": str(e)[:160]}
    return out


# --------------------------------------------------------------------------- analytics
def theme_of(tk: str, themes: dict) -> str:
    for th, members in themes.get("themes", {}).items():
        if tk in members:
            return th
    return "other"


def bucket_of(theme: str, themes: dict) -> str:
    for b, ths in themes.get("buckets", {}).items():
        if theme in ths:
            return b
    return "other"


def correlation_clusters(corr: pd.DataFrame, threshold: float) -> list[dict]:
    names = list(corr.columns)
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if corr.loc[a, b] >= threshold:
                parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    out = []
    for g in groups.values():
        if len(g) < 2:
            continue
        sub = corr.loc[g, g].values
        avg = (sub.sum() - len(g)) / (len(g) * (len(g) - 1))
        out.append({"tickers": sorted(g), "avg_pairwise_corr": round(float(avg), 2)})
    return sorted(out, key=lambda x: -len(x["tickers"]))


def load_watchlist(path: Path, held: list[str]) -> list[dict]:
    """Accepts a list of tickers / dicts, or {"candidates"|"watchlist"|"tickers": [...]}."""
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
        if tk in held or tk in seen:
            continue
        seen.add(tk)
        out.append({"ticker": tk, "sector": item.get("sector"), "type": item.get("type"),
                    "theme": item.get("theme"), "note": item.get("note")})
    return out


def _beta(a: pd.Series, b: pd.Series) -> float | None:
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(j) < 30:
        return None
    return float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var())


def _trend_stats(p: pd.Series) -> dict:
    p = p.dropna()
    n = len(p)
    if n < 2:
        return {"history_bars": int(n)}
    last = float(p.iloc[-1])
    out: dict = {"history_bars": int(n),
                 "ret_1d_pct": round((last / float(p.iloc[-2]) - 1) * 100, 2)}
    if n >= 6:
        out["ret_5d_pct"] = round((last / float(p.iloc[-6]) - 1) * 100, 2)
    if n < 6:
        return out
    y = p.tail(TRADING_DAYS + 1)
    out["ret_1m_pct"] = round((last / float(y.iloc[max(-22, -len(y))]) - 1) * 100, 1)
    out["ret_3m_pct"] = round((last / float(y.iloc[max(-64, -len(y))]) - 1) * 100, 1)
    out["from_52w_high_pct"] = round((last / float(y.max()) - 1) * 100, 1)
    if n >= TRADING_DAYS:
        out["ret_1y_pct"] = round((last / float(y.iloc[0]) - 1) * 100, 1)
        dd = y / y.cummax() - 1
        out["max_drawdown_1y_pct"] = round(float(dd.min() * 100), 1)
    if n >= 50:
        sma50 = float(p.tail(50).mean())
        out["sma50"] = round(sma50, 2)
        out["vs_sma50_pct"] = round((last / sma50 - 1) * 100, 1)
    if n >= 200:
        sma200 = float(p.tail(200).mean())
        out["sma200"] = round(sma200, 2)
        out["vs_sma200_pct"] = round((last / sma200 - 1) * 100, 1)
    return out


def _atr_stop(p: pd.Series) -> float | None:
    """Downside level from a close-to-close ATR proxy. High/low are not in the close frame."""
    p = p.dropna()
    if len(p) < 15:
        return None
    atr = float(p.diff().abs().tail(14).mean())
    px = float(p.iloc[-1])
    low20 = float(p.tail(20).min())
    dist = max(2.5 * atr, px - low20)
    if not math.isfinite(dist) or dist <= 0:
        return None
    return round(px - dist, 2)


def _earnings_info(tk: str, today: date) -> dict:
    import yfinance as yf
    try:
        frame = yf.Ticker(tk).get_earnings_dates(limit=12)
        index = getattr(frame, "index", None)
        if frame is None or index is None or len(index) == 0:
            return {}
        dates = []
        for idx in list(index):
            raw = idx.date() if hasattr(idx, "date") else None
            if isinstance(raw, datetime):
                raw = raw.date()
            if isinstance(raw, date):
                dates.append(raw)
        if not dates:
            return {}
        future = sorted(d for d in dates if d >= today)
        if not future:
            return {"last": max(dates).isoformat()}
        nxt = future[0]
        return {"date": nxt.isoformat(), "calendar_days": (nxt - today).days}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:160]}


def speaking_order(rnd: int, anchor: str) -> tuple[str, ...]:
    """Rotate who opens, so a one-round debate does not always end on neutral.

    The seat advances once per round and once per calendar day of the decision date.
    """
    roles = ("aggressive", "conservative", "neutral")
    try:
        ordinal = datetime.strptime(anchor, "%Y-%m-%d").date().toordinal()
    except ValueError:
        ordinal = date.today().toordinal()
    start = (ordinal + rnd - 1) % len(roles)
    return roles[start:] + roles[:start]


def _without_opus(model: str, allow_opus: bool, seat: str) -> str:
    if allow_opus or "opus" not in (model or "").lower():
        return model
    log(f"{seat} model {model} is an Opus model, which this add-on does not use by default (cost). "
        f"Using {QUICK_MODEL} instead. Pass --allow-opus to keep it.")
    return QUICK_MODEL


def build_stats(book: dict, close: pd.DataFrame, quotes: dict, themes: dict, targets: dict | None,
                meta: dict, corr_threshold: float, watchlist: list[dict], reports: dict,
                macro: dict | None, cash_policy: dict) -> dict:
    positions = book["positions"]
    cash = float(book.get("cash", 0))
    tickers = [p["ticker"] for p in positions]

    rows = []
    for p in positions:
        tk, qty, avg = p["ticker"], float(p["quantity"]), float(p["average_price"])
        px = quotes.get(tk, {}).get("price")
        mv = qty * px if px else None
        th = theme_of(tk, themes)
        rows.append({"ticker": tk, "quantity": qty, "avg_cost": avg, "price": px,
                     "price_source": quotes.get(tk, {}).get("source"),
                     "market_value": mv, "cost_basis": qty * avg,
                     "unrealized_pnl": (mv - qty * avg) if mv is not None else None,
                     "unrealized_pnl_pct": ((px / avg - 1) * 100) if px and avg else None,
                     "theme": th, "bucket": bucket_of(th, themes), **(meta.get(tk) or {})})
    invested = sum(r["market_value"] or 0 for r in rows)
    total = invested + cash
    for r in rows:
        r["weight_pct"] = (r["market_value"] or 0) / total * 100 if total else None

    rets_all = close.pct_change(fill_method=None).iloc[1:]
    win1 = rets_all.tail(TRADING_DAYS)          # 1y
    win2 = rets_all.tail(2 * TRADING_DAYS)      # 2y
    bench1 = win1[BENCH] if BENCH in win1 else None
    bench2 = win2[BENCH] if BENCH in win2 else None

    data_checks = []
    per = {}
    for tk in dict.fromkeys(tickers + [BENCH]):
        if tk not in win1:
            data_checks.append(f"{tk}: no price history from yfinance")
            continue
        s1 = win1[tk].dropna()
        ent = {"ann_vol_pct": round(float(s1.std() * math.sqrt(TRADING_DAYS) * 100), 1), "obs_1y": int(len(s1))}
        if bench1 is not None:
            s2 = win2[tk].dropna()
            b1 = _beta(s1, bench1)
            # A short history makes the 2y window the same sample as the 1y window. Do not copy it.
            b2 = _beta(s2, bench2) if len(s2) >= len(s1) + 40 else None
            ent["beta_1y"] = round(b1, 2) if b1 is not None else None
            ent["beta_2y"] = round(b2, 2) if b2 is not None else None
            ent["corr_spy_1y"] = round(float(win1[tk].corr(bench1)), 2)
            ent["corr_spy_2y"] = round(float(win2[tk].corr(bench2)), 2)
            # stress uses the more conservative (higher) of 1y / 2y beta
            ent["beta_spy"] = ent["beta_1y"]
            ent["stress_beta"] = round(max(x for x in (b1, b2, 0.0) if x is not None), 2)
            if b1 is not None and b2 is not None and abs(b1 - b2) >= 0.3:
                ent["beta_unstable"] = True
                data_checks.append(f"{tk}: 1y beta {b1:.2f} vs 2y beta {b2:.2f} - unstable; stress uses {ent['stress_beta']}")
        ent.update(_trend_stats(close[tk]))
        if tk in close:
            ent["atr_stop"] = _atr_stop(close[tk])
        per[tk] = ent
    for r in rows:
        r.update(per.get(r["ticker"], {}))
        kind = (r.get("type") or "").lower()
        if kind != "etf" and r["ticker"] != BENCH:
            earn = _earnings_info(r["ticker"], date.today())
            if earn:
                r["earnings"] = earn
                if earn.get("calendar_days") is not None and earn["calendar_days"] <= RISK_POLICY["earnings_warn_days"]:
                    data_checks.append(
                        f"{r['ticker']}: earnings {earn.get('date')} in {earn['calendar_days']} calendar days")
                elif earn.get("error"):
                    data_checks.append(f"{r['ticker']}: earnings calendar unavailable ({earn['error']})")

    # portfolio-level (1y window, current weights, cash earns 0)
    w = pd.Series({r["ticker"]: (r["market_value"] or 0) / total for r in rows})
    w = w[[t for t in w.index if t in win1.columns]]
    sub = win1[w.index].fillna(0.0)
    port_ret = pd.Series(sub.values @ w.values, index=sub.index)
    cov = sub.cov() * TRADING_DAYS
    port_vol = float(math.sqrt(max(w.values @ cov.values @ w.values, 0)))
    contrib = (w * (cov.values @ w.values)) / (port_vol ** 2) if port_vol else w * 0
    port_beta1 = _beta(port_ret, bench1) if bench1 is not None else None
    sub2 = win2[w.index].fillna(0.0)
    port_ret2 = pd.Series(sub2.values @ w.values, index=sub2.index)
    port_beta2 = _beta(port_ret2, bench2) if bench2 is not None else None
    nav = (1 + port_ret).cumprod()
    port_mdd = float((nav / nav.cummax() - 1).min())
    var95_1d = float(-np.percentile(port_ret, 5)) * total

    stress = {}
    for shock in (-0.05, -0.10, -0.20):
        loss = sum((r["market_value"] or 0) * (r.get("stress_beta") if r.get("stress_beta") is not None else 1.0) * shock
                   for r in rows)
        stress[f"SPY_{int(shock * 100)}pct_beta_scaled"] = {"pnl_usd": round(loss, 0),
                                                            "pnl_pct_of_account": round(loss / total * 100, 1)}
    if bench1 is not None:
        spy5 = (1 + bench1).rolling(5).apply(np.prod, raw=True) - 1
        loc = win1.index.get_loc(spy5.idxmin())
        span = win1.iloc[max(loc - 4, 0): loc + 1]
        replay = sum((r["market_value"] or 0) * (float((1 + span[r["ticker"]].fillna(0)).prod()) - 1)
                     for r in rows if r["ticker"] in span)
        stress["replay_worst_spy_5d_window"] = {
            "window": f"{span.index[0].date()} to {span.index[-1].date()}",
            "spy_move_pct": round(float(spy5.min()) * 100, 1),
            "pnl_usd": round(replay, 0), "pnl_pct_of_account": round(replay / total * 100, 1)}

    wts = np.array([r["weight_pct"] or 0 for r in rows]) / 100
    inv_w = wts / wts.sum() if wts.sum() else wts
    hhi = float((inv_w ** 2).sum())
    by_theme, by_bucket = {}, {}
    for r in rows:
        by_theme[r["theme"]] = by_theme.get(r["theme"], 0) + (r["weight_pct"] or 0)
        by_bucket[r["bucket"]] = by_bucket.get(r["bucket"], 0) + (r["weight_pct"] or 0)
    top3 = sorted(rows, key=lambda r: -(r["weight_pct"] or 0))[:3]
    corr = win1[[t for t in tickers if t in win1]].corr().round(2)

    drift = None
    tw = (targets or {}).get("weights_pct", {})
    if tw:
        drift = []
        for r in rows:
            t = tw.get(r["ticker"])
            if t is None or not r["price"]:
                continue
            drift.append({"ticker": r["ticker"], "current_pct": round(r["weight_pct"], 1), "target_pct": t,
                          "drift_pct": round(r["weight_pct"] - t, 1),
                          "shares_to_target": int(round((t / 100 * total - (r["market_value"] or 0)) / r["price"]))})
        drift.append({"ticker": "cash", "current_pct": round(cash / total * 100, 1),
                      "target_pct": f"dynamic ({cash_policy['min_pct']}-{cash_policy['max_pct']}%, soft default "
                                    f"{cash_policy['soft_default_pct']}%)",
                      "drift_pct": round(cash / total * 100 - cash_policy["soft_default_pct"], 1)})

    # market regime
    regime = {}
    if BENCH in close:
        regime.update({f"spy_{k}": v for k, v in _trend_stats(close[BENCH]).items()})
        regime["spy_realized_vol_20d_pct"] = round(float(win1[BENCH].tail(20).std() * math.sqrt(TRADING_DAYS) * 100), 1)
        regime["spy_realized_vol_1y_pct"] = per.get(BENCH, {}).get("ann_vol_pct")
    above200 = [r["ticker"] for r in rows if r.get("vs_sma200_pct") is not None and r["vs_sma200_pct"] > 0]
    above50 = [r["ticker"] for r in rows if r.get("vs_sma50_pct") is not None and r["vs_sma50_pct"] > 0]
    regime["holdings_above_sma200"] = f"{len(above200)}/{len(rows)}"
    regime["holdings_above_sma50"] = f"{len(above50)}/{len(rows)}"
    for sym, label in (("QQQ", "qqq"), ("SMH", "smh"), ("IWM", "iwm")):
        if sym in close:
            st = _trend_stats(close[sym])
            regime[f"{label}_ret_1d_pct"] = st.get("ret_1d_pct")
            regime[f"{label}_ret_5d_pct"] = st.get("ret_5d_pct")
    if "^VIX" in close:
        vx = close["^VIX"].dropna()
        if len(vx):
            age = (date.today() - vx.index[-1].date()).days
            regime["vix_yahoo"] = {"value": round(float(vx.iloc[-1]), 2), "date": str(vx.index[-1].date()),
                                   "age_days": age, "source": "yfinance ^VIX", "stale": age > 4}
            if age > 4:
                data_checks.append(f"^VIX is stale: {regime['vix_yahoo']['value']} on {regime['vix_yahoo']['date']} ({age} days)")
    if "^VIX3M" in close:
        v3 = close["^VIX3M"].dropna()
        if len(v3):
            age = (date.today() - v3.index[-1].date()).days
            regime["vix3m_yahoo"] = {"value": round(float(v3.iloc[-1]), 2), "date": str(v3.index[-1].date()),
                                     "age_days": age, "source": "yfinance ^VIX3M", "stale": age > 4}
    if macro:
        for k in ("vix", "10y_treasury", "2y_treasury", "10y_2y_spread", "fed_funds",
                  "core_cpi_index", "unemployment"):
            entry = macro.get(k)
            if isinstance(entry, dict) and "value" in entry:
                regime[k] = {"value": entry["value"], "date": entry.get("date"),
                             "age_days": entry.get("age_days"), "stale": bool(entry.get("stale"))}
                if entry.get("yoy_pct") is not None:
                    regime[k]["yoy_pct"] = entry["yoy_pct"]
                if entry.get("stale"):
                    data_checks.append(
                        f"{k} is stale: {entry['value']} as of {entry.get('date')} "
                        f"({entry.get('age_days')} days before {date.today().isoformat()})")
    ratings = {}
    for tk, rep in reports.items():
        ratings.setdefault(str(rep.get("pm_rating")), []).append(tk)
    regime["agent_ratings"] = ratings

    # watchlist candidates
    cands = []
    top_risk = contrib.sort_values(ascending=False).index[0] if len(contrib) else None
    for c in watchlist:
        tk = c["ticker"]
        ent = {**c, "price": quotes.get(tk, {}).get("price"), "has_agent_report": tk in reports,
               "agent_rating": reports.get(tk, {}).get("pm_rating") if tk in reports else None,
               **({k: v for k, v in (meta.get(tk) or {}).items() if k == "name"})}
        if tk not in win1 or win1[tk].dropna().empty:
            ent["error"] = "no price history"
            cands.append(ent)
            continue
        s = win1[tk]
        ent["ann_vol_pct"] = round(float(s.std() * math.sqrt(TRADING_DAYS) * 100), 1)
        if bench1 is not None:
            s2 = win2[tk].dropna()
            b1 = _beta(s.dropna(), bench1)
            b2 = _beta(s2, bench2) if len(s2) >= len(s.dropna()) + 40 else None
            ent["beta_1y"] = round(b1, 2) if b1 is not None else None
            ent["beta_2y"] = round(b2, 2) if b2 is not None else None
        ent["corr_to_book_1y"] = round(float(s.corr(port_ret)), 2)
        ent["avg_corr_to_holdings_1y"] = round(float(win1[list(w.index)].corrwith(s).mean()), 2)
        ent.update(_trend_stats(close[tk]))
        if tk in close:
            ent["atr_stop"] = _atr_stop(close[tk])
        if (ent.get("type") or "").lower() != "etf":
            earn = _earnings_info(tk, date.today())
            if earn:
                ent["earnings"] = earn
                if earn.get("calendar_days") is not None and earn["calendar_days"] <= RISK_POLICY["earnings_warn_days"]:
                    data_checks.append(f"{tk}: earnings {earn.get('date')} in {earn['calendar_days']} calendar days")
        # diversification benefit: add 3% funded from cash, and swap 3% out of top risk contributor
        x = s.fillna(0.0)
        for label, w_new in (("add3pct_from_cash", w.copy()), (f"swap3pct_from_{top_risk}", w.copy())):
            if label.startswith("swap") and top_risk:
                w_new[top_risk] = max(w_new[top_risk] - 0.03, 0)
            mat = pd.concat([sub, x.rename(tk)], axis=1).fillna(0.0)
            ww = np.append(w_new.values, 0.03)
            vol_new = float(math.sqrt(max(ww @ (mat.cov().values * TRADING_DAYS) @ ww, 0)))
            ent[f"port_vol_change_pp_{label}"] = round((vol_new - port_vol) * 100, 2)
        cands.append(ent)

    return {
        "as_of_prices": str(close.index[-1].date()),
        "account": {"total_value": round(total, 2), "invested": round(invested, 2), "cash": cash,
                    "cash_pct": round(cash / total * 100, 1) if total else None,
                    "unrealized_pnl": round(sum(r["unrealized_pnl"] or 0 for r in rows), 2)},
        "cash_policy": cash_policy,
        "positions": rows,
        "risk": {"portfolio_ann_vol_pct": round(port_vol * 100, 1),
                 "portfolio_beta_spy_1y": round(port_beta1, 2) if port_beta1 is not None else None,
                 "portfolio_beta_spy_2y": round(port_beta2, 2) if port_beta2 is not None else None,
                 "spy_ann_vol_pct": per.get(BENCH, {}).get("ann_vol_pct"),
                 "max_drawdown_1y_current_weights_pct": round(port_mdd * 100, 1),
                 "var95_1d_usd": round(var95_1d, 0),
                 "risk_contribution_pct": {k: round(float(v) * 100, 1) for k, v in contrib.sort_values(ascending=False).items()}},
        "stress": stress,
        "concentration": {"hhi_invested": round(hhi, 3), "effective_positions": round(1 / hhi, 1) if hhi else None,
                          "top3": [{"ticker": r["ticker"], "weight_pct": round(r["weight_pct"], 1)} for r in top3],
                          "by_theme_pct": {k: round(v, 1) for k, v in by_theme.items()},
                          "by_bucket_pct": {k: round(v, 1) for k, v in by_bucket.items()}},
        "regime": regime,
        "risk_policy": RISK_POLICY,
        "correlation_matrix": corr.to_dict(),
        "correlation_clusters": correlation_clusters(corr.drop(index=BENCH, columns=BENCH, errors="ignore"), corr_threshold),
        "correlation_threshold": corr_threshold,
        "target_drift": drift,
        "watchlist": cands,
        "watchlist_swap_source": top_risk,
        "data_checks": data_checks,
    }


# --------------------------------------------------------------------------- prompts & schemas
FM_SYSTEM = """You are the senior Fund Manager / CIO sitting on top of a TradingAgents desk.
For each holding (and any watchlist candidate with a report) a team of analysts, bull/bear researchers,
a trader, a risk debate and a Portfolio Manager has produced a single-stock decision. Those PMs judged ONE
stock at a time and did NOT see portfolio weights, correlations or concentration. You make the
whole-portfolio decision they could not.

Rules:
1. Diagnose the portfolio from the quantitative stats (weights, P&L, vol, 1y/2y beta, risk contribution,
   correlation clusters, VaR, stress, concentration, drift) plus the market regime and macro snapshot.
2. For EVERY holding decide keep / trim / exit / add, with exact WHOLE-SHARE trades. Never sell more than
   held. No shorting, options or margin; ending cash >= 0. Use the provided prices.
3. Weigh each single-stock PM view against portfolio context. You may size differently or override; say so.
4. Owner target weights are GUIDANCE. Cash is NOT fixed: choose a target cash % each run inside the
   cash_policy band, from the market regime (VIX, SPY vs SMA50/200, realized vol, drawdown), FRED macro,
   stress results, and the opportunity set (agent ratings, candidate quality). The soft default is only a
   starting point. Justify the level and give concrete conditions to deploy or raise cash. Your trades must
   actually land near your stated cash target (within ~3 points) unless you explain staged deployment.
5. Consider swaps: whether freed cash is better used in watchlist candidates (stocks or sector ETFs) than
   kept or recycled into holdings. Use candidates' trend, vol, beta, correlation to the book and the
   diversification numbers (port_vol_change_pp_*). A candidate without an agent report can still be bought,
   but say its rating is missing and size it smaller / stage it; prefer diversified ETFs when evidence is thin.
6. Prioritise execution (risk reduction before adds). Give a numeric stop_or_reassess for every name you keep.
   If you omit one, code fills the supplied atr_stop and warns. atr_stop is a close-to-close ATR proxy, not a hard stop.
7. State what evidence would change your mind. Be concrete, numeric, honest about data gaps.
8. `risk_policy` is enforced in code after your plan (it does not rewrite share counts). Stay inside it:
   single-stock cap, a tighter cap when volatility is high or history is short, broad-index cap,
   top-3 cap, minimum trade size, and the AI sleeve cap. The AI sleeve is ai_growth plus AAPL/MSFT/GOOG
   plus spy_tech_share of SPY, because those megacaps are tagged quality and SPY is about one-third tech.
   Flag earnings inside earnings_warn_days rather than adding into them. Regime values carry their dates;
   a stale flag means the print is older than one session (or one release for monthly series).
This is decision support, not financial advice."""

ANALYST_ROLES = {
    "aggressive": """You are the AGGRESSIVE portfolio risk analyst on the fund's risk committee. You champion
upside capture and challenge over-de-risking: excessive cash drag, selling quality winners into weakness,
trimming too much of strong trends, under-using the opportunity set. You still respect hard limits
(no margin/shorts) and must use the numbers.""",
    "conservative": """You are the CONSERVATIVE portfolio risk analyst on the fund's risk committee. Your priority
is capital preservation: tail and gap risk, correlation spikes in stress, concentration, beta, drawdown,
liquidity, earnings events, macro/rate risk, and data reliability. Push for more protection where the
numbers justify it, but don't argue for cash for its own sake.""",
    "neutral": """You are the NEUTRAL portfolio risk analyst on the fund's risk committee. Weigh the aggressive and
conservative arguments against the numbers, call out where either overreaches, and propose the balanced
adjustments that best improve risk-adjusted outcome.""",
}
ANALYST_TASK = """Critique the Fund Manager's DRAFT plan using the whole-portfolio stats (correlation clusters,
1y/2y beta, VaR, risk contribution, stress tests, concentration, target drift, regime, macro) and the
watchlist candidates. Debate the CASH level explicitly (give your own target cash %, inside the band).
Respond directly to the other analysts' earlier points where present. Propose specific whole-share changes
(ticker, action, shares) or say 'no change'. Be concise and numeric (<= ~500 words of content)."""


def _obj(props: dict, req: list | None = None) -> dict:
    return {"type": "object", "additionalProperties": False, "required": req or list(props), "properties": props}


S, N, I = {"type": "string"}, {"type": "number"}, {"type": "integer"}
NN = {"type": ["number", "null"]}
ARR_S = {"type": "array", "items": S}

TRADE_ITEM = _obj({
    "ticker": S,
    "is_holding": {"type": "boolean"},
    "decision": {"type": "string", "enum": ["keep", "trim", "exit", "add", "new_position", "pass"]},
    "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
    "shares": {**I, "description": "whole shares; 0 for HOLD"},
    "est_price": N, "order_type": {"type": "string", "enum": ["market", "limit", "stop", "staged", "none"]},
    "limit_price": NN, "timing": S, "agent_rating": S,
    "follows_agent": {"type": "string", "enum": ["agrees", "sizes_differently", "overrides", "no_agent_report"]},
    "rationale": S,
})
CASH_DECISION = _obj({
    "target_cash_pct": N, "regime": {"type": "string", "enum": ["risk_on", "neutral", "cautious", "risk_off", "crisis"]},
    "regime_signals": ARR_S, "rationale": S, "deploy_conditions": ARR_S, "raise_conditions": ARR_S,
})
PLAN_PROPS = {
    "headline": S, "diagnosis": S, "cash_decision": CASH_DECISION,
    "key_risks": {"type": "array", "items": _obj({"risk": S, "severity": {"type": "string", "enum": ["high", "medium", "low"]}, "detail": S})},
    "correlation_clusters": {"type": "array", "items": _obj({"name": S, "tickers": ARR_S, "comment": S})},
    "decision_summary": S,
    "trades": {"type": "array", "items": TRADE_ITEM},
    "swaps_considered": {"type": "array", "items": _obj({"from": S, "to": S, "verdict": {"type": "string", "enum": ["do_now", "staged", "watch", "reject"]}, "reason": S})},
    "priority_order": ARR_S,
    "cash_plan": _obj({"summary": S, "deployments": {"type": "array", "items": _obj({"use": S, "amount_usd": N, "condition": S})}}),
    "levels": {"type": "array", "items": _obj({"ticker": S, "stop_or_reassess": NN, "add_or_reclaim": NN, "note": S})},
    "what_would_change_my_mind": ARR_S, "data_gaps": ARR_S,
    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
}
DRAFT_SCHEMA = _obj(PLAN_PROPS)
FINAL_SCHEMA = _obj({
    **PLAN_PROPS,
    "debate_takeaways": {"type": "array", "items": _obj({"analyst": S, "point": S, "accepted": {"type": "string", "enum": ["accepted", "partly", "rejected"]}, "response": S})},
    "changes_vs_draft": {"type": "array", "items": _obj({"item": S, "draft": S, "final": S, "reason": S})},
})
ANALYST_SCHEMA = _obj({
    "stance_summary": S,
    "cash_view": _obj({"target_cash_pct": N, "rationale": S}),
    "key_points": ARR_S,
    "proposed_changes": {"type": "array", "items": _obj({"ticker": S, "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD", "NO_CHANGE"]}, "shares": I, "reason": S})},
    "rebuttals": ARR_S,
    "risks_flagged": ARR_S,
})


def build_context(book, targets, stats, reports, macro, as_of) -> str:
    slim_pos = [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "price_source"}
                for r in stats["positions"]]
    stats_view = {k: v for k, v in stats.items() if k not in ("positions", "watchlist")}
    parts = [
        f"# Fund review - decision date {as_of}",
        "## Book (shares, average cost, cash)", json.dumps(book, indent=1),
        "## Owner target weights (GUIDANCE ONLY; cash entry is a soft default, cash target is dynamic)",
        json.dumps(targets, indent=1) if targets else "none provided",
        "## Positions with live prices and per-stock stats", json.dumps(slim_pos, indent=1),
        "## Portfolio-level quantitative stats, regime and cash policy", json.dumps(stats_view, indent=1),
        "## Macro snapshot (FRED, latest values)", json.dumps(macro, indent=1) if macro else "not fetched",
        "## Watchlist candidates (NOT held)",
        json.dumps(stats["watchlist"], indent=1) if stats["watchlist"] else "none",
        "## Single-stock agent decisions (latest TradingAgents run per ticker)",
    ]
    for tk, r in reports.items():
        tag = "HOLDING" if tk in {p["ticker"] for p in book["positions"]} else "CANDIDATE"
        parts.append(f"### [{tag}] {tk} - PM rating: {r['pm_rating']} | trader: {r['trader_action']} "
                     f"| analysis date {r['analysis_date']} | report {r['report_dir']}")
        parts.append(f"Trader position sizing: {r['trader_position_sizing']}")
        parts.append("Portfolio Manager decision:\n" + (r["pm_decision"] or "(missing)"))
        if r.get("research_manager"):
            parts.append("Research Manager plan:\n" + r["research_manager"])
        if r.get("risk_debate"):
            for k, v in r["risk_debate"].items():
                parts.append(f"Risk ({k}):\n{v}")
    return "\n\n".join(parts)


def call_llm(model: str, system: str, user: str, schema: dict, name: str, max_tokens: int,
             effort: str | None, label: str) -> dict:
    from openai import OpenAI

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set in the container environment (.env)")
    client = OpenAI(base_url=OPENROUTER_BASE, api_key=key, timeout=900, max_retries=2)
    extra = {"usage": {"include": True}}
    if effort:
        extra["reasoning"] = {"effort": effort}
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    attempts = [("json_schema", {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}),
                ("json_object", {"type": "json_object"}), ("text", None)]
    last_err = None
    for mode, rf in attempts:
        try:
            t0 = time.time()
            kwargs = dict(model=model, messages=msgs, max_tokens=max_tokens, extra_body=extra)
            if rf:
                kwargs["response_format"] = rf
            if mode == "json_object":
                kwargs["messages"] = [msgs[0], {"role": "user", "content": user + "\n\nReturn ONLY JSON matching this schema:\n" + json.dumps(schema)}]
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            usage = resp.usage.model_dump() if resp.usage else {}
            parsed = None
            if mode != "text":
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    m = re.search(r"\{.*\}", content, re.S)
                    parsed = json.loads(m.group(0)) if m else None
                if parsed is None:
                    raise ValueError("model did not return valid JSON")
            out = {"label": label, "mode": mode, "content": content, "parsed": parsed, "usage": usage,
                   "seconds": round(time.time() - t0, 1), "model": getattr(resp, "model", model)}
            log(f"  {label}: {model} done in {out['seconds']}s ({mode}), tokens {usage.get('prompt_tokens')}/"
                f"{usage.get('completion_tokens')}, cost {usage.get('cost', 'n/a')}")
            return out
        except Exception as e:  # noqa: BLE001
            last_err = e
            log(f"  {label}: mode '{mode}' failed: {str(e)[:300]}")
    raise RuntimeError(f"{label}: all LLM attempts failed: {last_err}")


def run_debate(context: str, draft: dict, models: dict, rounds: int, max_tokens: int, effort: str | None,
               anchor: str) -> list[dict]:
    transcript = []
    for rnd in range(1, rounds + 1):
        order = speaking_order(rnd, anchor)
        log(f"  risk round {rnd} speaking order: {' -> '.join(order)}")
        for role in order:
            prior = "\n\n".join(
                f"[Round {t['round']}] {t['role'].upper()} ({t['model']}):\n" + json.dumps(t["parsed"] or t["content"], indent=1)
                for t in transcript) or "(you speak first this round)"
            user = (context + "\n\n## Fund Manager DRAFT plan\n" + json.dumps(draft, indent=1)
                    + f"\n\n## Risk committee debate so far (round {rnd} of {rounds})\n" + prior
                    + "\n\n## Your task\n" + ANALYST_TASK)
            res = call_llm(models[role], ANALYST_ROLES[role], user, ANALYST_SCHEMA, f"{role}_risk_view",
                           max_tokens, effort, f"risk/{role} r{rnd}")
            transcript.append({"round": rnd, "role": role, "model": models[role], "parsed": res["parsed"],
                               "content": res["content"] if not res["parsed"] else None,
                               "usage": res["usage"], "seconds": res["seconds"], "mode": res["mode"]})
    return transcript


# --------------------------------------------------------------------------- guardrails
def _info_for(tk: str, pos: dict, wl: dict) -> dict:
    return pos.get(tk) or wl.get(tk) or {}


def _beta_of(info: dict) -> tuple[float, str]:
    for key in ("stress_beta", "beta_1y", "beta_spy"):
        v = info.get(key)
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v), key
    return 1.0, "assumed 1.0 (no beta)"


def _position_cap(info: dict, policy: dict) -> tuple[float, str]:
    theme = info.get("theme")
    kind = (info.get("type") or "").lower()
    if theme == "core_index" or (kind == "etf" and info.get("ticker") in (None, BENCH)) or info.get("ticker") == BENCH:
        return policy["broad_index_cap_pct"], "broad index"
    obs = info.get("obs_1y") or info.get("history_bars") or 0
    if obs and obs < policy["short_history_bars"]:
        return policy["short_history_cap_pct"], f"history {obs} bars"
    vol = info.get("ann_vol_pct")
    if isinstance(vol, (int, float)) and vol > policy["high_vol_threshold_pct"]:
        return policy["high_vol_cap_pct"], f"vol {vol}%"
    return policy["single_stock_cap_pct"], "single name"


def _stop_number(value) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
        return float(value)
    return None


def validate_trades(decision: dict, stats: dict) -> dict:
    pos = {r["ticker"]: r for r in stats["positions"]}
    wl = {c["ticker"]: c for c in stats.get("watchlist", [])}
    cash = float(stats["account"]["cash"])
    total = float(stats["account"]["total_value"] or 0)
    pol = stats["cash_policy"]
    policy = {**RISK_POLICY, **(stats.get("risk_policy") or {})}
    warnings, rows = [], []
    new_qty = {tk: r["quantity"] for tk, r in pos.items()}
    prices = {}
    seen = set()
    min_trade = float(policy["min_trade_usd"])
    for t in decision.get("trades", []):
        tk = t.get("ticker")
        info = _info_for(tk, pos, wl)
        if info.get("price"):
            px = float(info["price"])
        else:
            px = float(t.get("est_price") or 0)
            if tk not in pos:
                warnings.append(f"{tk}: not in book or watchlist - priced at model estimate ${px:,.2f}")
        prices[tk] = px
        sh = int(t.get("shares") or 0)
        act = (t.get("action") or "HOLD").upper()
        if tk in seen:
            warnings.append(f"{tk}: duplicate trade ignored (first row kept)")
            continue
        if act == "HOLD" and sh:
            warnings.append(f"{tk}: HOLD with {sh} shares; treated as no trade")
            sh = 0
        if act == "SELL":
            held = new_qty.get(tk, 0)
            if sh > held:
                warnings.append(f"{tk}: model asked to sell {sh} but only {held:g} held - capped")
                sh = int(held)
            new_qty[tk] = held - sh
            cash += sh * px
        elif act == "BUY":
            new_qty[tk] = new_qty.get(tk, 0) + sh
            cash -= sh * px
        if act in ("BUY", "SELL") and sh > 0 and sh * px < min_trade:
            warnings.append(f"{tk}: {act} {sh} is ${sh * px:,.0f}, below the ${min_trade:,.0f} minimum trade")
        decision_name = (t.get("decision") or "").lower()
        if decision_name in ("trim", "exit") and act != "SELL":
            warnings.append(f"{tk}: decision {decision_name} but action is {act}")
        if decision_name in ("add", "new_position") and act != "BUY":
            warnings.append(f"{tk}: decision {decision_name} but action is {act}")
        if (t.get("order_type") or "").lower() == "staged" and act in ("BUY", "SELL") and sh:
            warnings.append(f"{tk}: staged order is counted as filled in the post-trade cash and risk figures")
        seen.add(tk)
        rows.append({**t, "shares": sh, "value_usd": round(sh * px, 2), "calc_price": round(px, 2)})
    missing = [tk for tk in pos if tk not in seen]
    if missing:
        warnings.append(f"no instruction for: {', '.join(missing)} (treated as HOLD)")
    if cash < 0:
        warnings.append(f"trades overspend cash by ${-cash:,.0f} - buys must be reduced")
    cash_pct = cash / total * 100 if total else 0
    tgt = (decision.get("cash_decision") or {}).get("target_cash_pct")
    if tgt is not None:
        if not (pol["min_pct"] <= tgt <= pol["max_pct"]):
            warnings.append(f"target cash {tgt}% is outside the policy band {pol['min_pct']}-{pol['max_pct']}%")
        if abs(cash_pct - tgt) > 3:
            warnings.append(f"post-trade cash {cash_pct:.1f}% differs from stated target {tgt}% by more than 3 points")

    levels = {}
    for lv in decision.get("levels") or []:
        if isinstance(lv, dict) and lv.get("ticker") and lv["ticker"] not in levels:
            levels[lv["ticker"]] = lv
    after = []
    beta_w = 0.0
    ai = 0.0
    assumed = []
    for tk, q in new_qty.items():
        info = _info_for(tk, pos, wl)
        px = prices.get(tk)
        if px is None:
            px = float(info.get("price") or 0)
        value = q * (px or 0)
        weight = value / total * 100 if total else 0
        beta, beta_src = _beta_of(info)
        if beta_src.startswith("assumed") and q:
            assumed.append(tk)
        beta_w += (value / total) * beta if total else 0
        cap, why = _position_cap({**info, "ticker": tk}, policy)
        if weight > cap + 0.05 and q:
            warnings.append(f"{tk}: post-trade weight {weight:.1f}% exceeds the {cap:.0f}% cap ({why})")
        theme = info.get("theme")
        if theme == "ai_growth" or tk in AI_MEGACAP:
            ai += weight
        elif tk == BENCH:
            ai += weight * float(policy["spy_tech_share"])
        downside, downside_src = None, None
        if q:
            stop = _stop_number((levels.get(tk) or {}).get("stop_or_reassess"))
            if stop is None:
                fallback = _stop_number(info.get("atr_stop"))
                if fallback is not None:
                    downside, downside_src = fallback, "atr_fallback"
                    warnings.append(
                        f"{tk}: no stop/downside in the plan; ATR fallback ${fallback:,.2f} "
                        "(2.5x 14-day close-to-close ATR, and not above a break of the 20-day low)")
                else:
                    warnings.append(f"{tk}: no stop/downside and no ATR fallback (not enough history)")
            else:
                downside, downside_src = stop, "plan"
            earn = info.get("earnings") or {}
            days = earn.get("calendar_days")
            if days is not None and days <= policy["earnings_warn_days"]:
                warnings.append(
                    f"{tk}: earnings {earn.get('date')} in {days} calendar days "
                    f"(warning window is {policy['earnings_warn_days']} days)")
        after.append({"ticker": tk, "shares_after": q, "value_after": round(value, 2),
                      "weight_after_pct": round(weight, 1), "is_new": tk not in pos,
                      "beta": round(beta, 2), "beta_source": beta_src,
                      "downside": downside, "downside_source": downside_src})
    if assumed:
        warnings.append("post-trade beta uses 1.0 for: " + ", ".join(assumed))
    if ai > policy["ai_sleeve_cap_pct"] + 0.05:
        warnings.append(
            f"AI look-through {ai:.1f}% exceeds the {policy['ai_sleeve_cap_pct']:.0f}% sleeve cap "
            f"(ai_growth + AAPL/MSFT/GOOG + {float(policy['spy_tech_share']):.0%} of SPY)")
    ranked = sorted((a for a in after if a["shares_after"]), key=lambda a: -a["weight_after_pct"])
    top3 = sum(a["weight_after_pct"] for a in ranked[:3])
    if top3 > policy["top3_cap_pct"] + 0.05:
        names = ", ".join(a["ticker"] for a in ranked[:3])
        warnings.append(f"top 3 ({names}) are {top3:.1f}% of NAV, above the {policy['top3_cap_pct']:.0f}% cap")
    shocks = {}
    for shock in (-0.05, -0.10, -0.20):
        pnl = 0.0
        for a in after:
            if not a["shares_after"]:
                continue
            pnl += a["value_after"] * a["beta"] * shock
        shocks[f"SPY_{int(shock * 100)}pct_beta_scaled"] = {
            "pnl_usd": round(pnl, 0), "pnl_pct_of_account": round(pnl / total * 100, 1) if total else None}
    after.append({"ticker": "cash", "shares_after": None, "value_after": round(cash, 2),
                  "weight_after_pct": round(cash_pct, 1), "is_new": False,
                  "beta": 0.0, "beta_source": "cash", "downside": None, "downside_source": None})
    post_trade_risk = {"beta_spy": round(beta_w, 2), "ai_lookthrough_pct": round(ai, 1),
                       "top3_pct": round(top3, 1), "stress": shocks,
                       "note": "Beta and stress are share-weighted from each name's stress beta (else 1y beta, else 1.0). Cash beta is 0. Not a covariance recomputation."}
    return {"trades": rows, "post_trade": after, "ending_cash": round(cash, 2),
            "ending_cash_pct": round(cash_pct, 1), "warnings": warnings,
            "post_trade_risk": post_trade_risk}


def diff_plans(draft: dict | None, final: dict | None) -> list[dict]:
    if not draft or not final:
        return []

    def m(p):
        return {t["ticker"]: (t["action"], int(t.get("shares") or 0)) for t in p.get("trades", [])}
    a, b = m(draft), m(final)
    out = []
    for tk in dict.fromkeys(list(a) + list(b)):
        da, fa = a.get(tk, ("HOLD", 0)), b.get(tk, ("HOLD", 0))
        if da != fa:
            out.append({"ticker": tk, "draft": f"{da[0]} {da[1]}", "final": f"{fa[0]} {fa[1]}"})
    dc = (draft.get("cash_decision") or {}).get("target_cash_pct")
    fc = (final.get("cash_decision") or {}).get("target_cash_pct")
    if dc != fc:
        out.append({"ticker": "target cash %", "draft": str(dc), "final": str(fc)})
    return out


# --------------------------------------------------------------------------- render
def fmt_money(x):
    return "n/a" if x is None else f"${x:,.0f}"


def _cell(x, n=220):
    return _clip(str(x) if x is not None else "", n)


def render_plan_trades(L, plan, check, targets):
    order = {tk: i for i, tk in enumerate(plan.get("priority_order", []))}
    trades = sorted(check["trades"], key=lambda t: order.get(t["ticker"], 99))
    L.append("| # | Ticker | Decision | Action | Shares | Est. price | Value | Order | Limit | Timing | Agent rating | vs agent | Rationale |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, t in enumerate(trades, 1):
        L.append(f"| {i} | {t['ticker']}{'' if t.get('is_holding', True) else ' (new)'} | {t.get('decision', '')} | "
                 f"{t['action']} | {t['shares']} | ${t['calc_price']:,.2f} | {fmt_money(t['value_usd'])} | "
                 f"{t.get('order_type', '')} | {'' if t.get('limit_price') in (None, 0) else t['limit_price']} | "
                 f"{_cell(t.get('timing'), 160)} | {t.get('agent_rating', '')} | {t.get('follows_agent', '')} | "
                 f"{_cell(t.get('rationale'), 400)} |")
    L.append(f"\nPriority order: {' -> '.join(plan.get('priority_order', []))}\n")
    L.append("### Post-trade check (computed by the add-on, not the model)")
    L.append(f"Ending cash: **{fmt_money(check['ending_cash'])} ({check['ending_cash_pct']}%)**")
    risk = check.get("post_trade_risk") or {}
    if risk:
        L.append(f"Post-trade beta (code): **{risk.get('beta_spy')}** | AI look-through **{risk.get('ai_lookthrough_pct')}%** "
                 f"(ai_growth + AAPL/MSFT/GOOG + {int(SPY_TECH_SHARE * 100)}% of SPY) | top 3 **{risk.get('top3_pct')}%**")
        L.append("Post-trade stress (code, beta-scaled): " + "; ".join(
            f"{k}: {fmt_money(v['pnl_usd'])} ({v['pnl_pct_of_account']}%)" for k, v in (risk.get("stress") or {}).items()))
        if risk.get("note"):
            L.append(risk["note"])
    stops = [a for a in check.get("post_trade", []) if a.get("downside")]
    if stops:
        L.append("Downside levels: " + "; ".join(
            f"{a['ticker']} ${a['downside']} ({a.get('downside_source')})" for a in stops))
    if check["warnings"]:
        L.append("\nWarnings:\n" + "\n".join(f"- {w}" for w in check["warnings"]))
    tw = (targets or {}).get("weights_pct", {})
    L.append("\n| Ticker | Shares after | Value after | Weight after | Target (guidance) |\n|---|---|---|---|---|")
    for a in check["post_trade"]:
        sh = "" if a["shares_after"] is None else f"{a['shares_after']:g}"
        tg = "dynamic" if a["ticker"] == "cash" else (f"{tw[a['ticker']]}%" if a["ticker"] in tw else "")
        L.append(f"| {a['ticker']}{' (new)' if a['is_new'] else ''} | {sh} | {fmt_money(a['value_after'])} | {a['weight_after_pct']}% | {tg} |")


def render_markdown(as_of, models, stats, reports, macro, calls, draft, debate, final, check_d, check_f, targets) -> str:
    L = []
    acct = stats["account"]
    pol = stats["cash_policy"]
    plan = final or draft
    check = check_f or check_d
    L.append(f"# Fund Manager portfolio report - {as_of}")
    L.append(f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC | fund manager `{models['fm']}` | "
             f"risk team: aggressive `{models['aggressive']}`, conservative `{models['conservative']}`, neutral "
             f"`{models['neutral']}` | prices as of {stats['as_of_prices']} | decision support, not financial advice._\n")
    if models.get("note"):
        L.append(f"_{models['note']}_\n")
    if not plan:
        L.append("_LLM step skipped (--no-llm). Quantitative snapshot only._\n")
    else:
        stage = "FINAL (after risk-committee debate)" if final else "DRAFT (no debate run)"
        L.append(f"## Verdict - {stage}\n**{plan.get('headline', '')}**  (confidence: {plan.get('confidence', 'n/a')})\n")
        cd = plan.get("cash_decision") or {}
        L.append("## Cash decision")
        L.append(f"> **Target cash: {cd.get('target_cash_pct')}%** (policy band {pol['min_pct']}-{pol['max_pct']}%, soft default "
                 f"{pol['soft_default_pct']}%) | regime: **{cd.get('regime')}** | post-trade cash: "
                 f"**{fmt_money(check['ending_cash'])} ({check['ending_cash_pct']}%)** | current {acct['cash_pct']}%")
        if final and draft:
            L.append(f"> Draft target was {(draft.get('cash_decision') or {}).get('target_cash_pct')}%; risk team views: "
                     + ", ".join(f"{t['role']} {((t['parsed'] or {}).get('cash_view') or {}).get('target_cash_pct')}%"
                                 for t in debate if t.get("round") == max(x['round'] for x in debate)))
        L.append(f"\n{cd.get('rationale', '')}\n")
        if cd.get("regime_signals"):
            L.append("Regime signals: " + "; ".join(cd["regime_signals"]))
        L.append("\n**Deploy cash when:**\n" + "\n".join(f"- {x}" for x in cd.get("deploy_conditions", [])))
        L.append("\n**Raise cash when:**\n" + "\n".join(f"- {x}" for x in cd.get("raise_conditions", [])) + "\n")
        L.append(f"## Diagnosis\n{plan.get('diagnosis', '')}\n")
        L.append(f"## Decision\n{plan.get('decision_summary', '')}\n")
        if final:
            L.append("## What changed vs the draft")
            dd = diff_plans(draft, final)
            if dd:
                L.append("| Item | Draft | Final |\n|---|---|---|")
                L += [f"| {x['ticker']} | {x['draft']} | {x['final']} |" for x in dd]
            else:
                L.append("_No change to trades or cash target._")
            if final.get("changes_vs_draft"):
                L.append("\nManager's explanation:")
                L += [f"- **{c['item']}**: {c['draft']} -> {c['final']}. {c['reason']}" for c in final["changes_vs_draft"]]
            if final.get("debate_takeaways"):
                L.append("\nDebate points and the manager's response:")
                L += [f"- [{t['analyst']}] {t['point']} - **{t['accepted']}**: {t['response']}" for t in final["debate_takeaways"]]
            L.append("")
        L.append("## Final trades (whole shares)" if final else "## Trades (whole shares)")
        render_plan_trades(L, plan, check, targets)
        if plan.get("swaps_considered"):
            L.append("\n## Swaps considered (holdings -> watchlist)")
            L.append("| From | To | Verdict | Reason |\n|---|---|---|---|")
            L += [f"| {s['from']} | {s['to']} | {s['verdict']} | {_cell(s['reason'], 300)} |" for s in plan["swaps_considered"]]
        cp = plan.get("cash_plan", {})
        L.append(f"\n## Where the freed cash goes\n{cp.get('summary', '')}")
        for dep in cp.get("deployments", []):
            L.append(f"- {fmt_money(dep.get('amount_usd'))} -> {dep.get('use')} (when: {dep.get('condition')})")
        L.append("\n## Key risks")
        L += [f"- **[{r.get('severity')}] {r.get('risk')}** - {r.get('detail')}" for r in plan.get("key_risks", [])]
        L.append("\n## Correlation clusters (manager view)")
        L += [f"- **{c.get('name')}**: {', '.join(c.get('tickers', []))} - {c.get('comment')}" for c in plan.get("correlation_clusters", [])]
        L.append("\n## Stops and levels\n| Ticker | Stop / reassess | Add / reclaim | Note |\n|---|---|---|---|")
        L += [f"| {lv.get('ticker')} | {lv.get('stop_or_reassess') or ''} | {lv.get('add_or_reclaim') or ''} | {_cell(lv.get('note'), 300)} |"
              for lv in plan.get("levels", [])]
        L.append("\n## What would change my mind")
        L += [f"- {x}" for x in plan.get("what_would_change_my_mind", [])]
        if plan.get("data_gaps"):
            L.append("\n## Data gaps")
            L += [f"- {x}" for x in plan["data_gaps"]]

    if debate:
        L.append("\n---\n## Portfolio risk committee debate")
        for t in debate:
            p = t.get("parsed") or {}
            L.append(f"\n### Round {t['round']} - {t['role'].capitalize()} (`{t['model']}`)")
            if not p:
                L.append(t.get("content") or "")
                continue
            L.append(f"**Stance:** {p.get('stance_summary', '')}\n")
            cv = p.get("cash_view") or {}
            L.append(f"**Cash view:** {cv.get('target_cash_pct')}% - {cv.get('rationale', '')}\n")
            L += [f"- {k}" for k in p.get("key_points", [])]
            ch = [c for c in p.get("proposed_changes", []) if c.get("action") != "NO_CHANGE"]
            if ch:
                L.append("\nProposed changes: " + "; ".join(f"{c['action']} {c['shares']} {c['ticker']} ({_cell(c['reason'], 160)})" for c in ch))
            if p.get("rebuttals"):
                L.append("\nRebuttals: " + " / ".join(p["rebuttals"]))
        if draft and final:
            L.append("\n### Draft plan trades (before debate)")
            L.append("| Ticker | Action | Shares | Decision |\n|---|---|---|---|")
            L += [f"| {t['ticker']} | {t['action']} | {t['shares']} | {t.get('decision', '')} |" for t in check_d["trades"]]
            L.append(f"\nDraft headline: {draft.get('headline', '')} | draft ending cash {fmt_money(check_d['ending_cash'])} ({check_d['ending_cash_pct']}%)")

    # side-by-side per-stock agent views
    L.append("\n---\n## Per-stock agent views (side by side)")
    fin_map = {t["ticker"]: t for t in (check["trades"] if check else [])}
    L.append("| Ticker | Bull | Bear | Research mgr | Trader | Risk team (agg / cons / neutral) | PM rating | Fund manager |")
    L.append("|---|---|---|---|---|---|---|---|")
    for tk, r in reports.items():
        g = r.get("gists", {})
        fm = fin_map.get(tk)
        fm_txt = f"{fm['action']} {fm['shares']} ({fm.get('decision', '')})" if fm else ""
        L.append(f"| {tk} | {_cell(g.get('bull'), 200)} | {_cell(g.get('bear'), 200)} | {g.get('research_manager') or ''} | "
                 f"{r.get('trader_action') or ''} | A: {_cell(g.get('risk_aggressive'), 150)}<br>C: {_cell(g.get('risk_conservative'), 150)}"
                 f"<br>N: {_cell(g.get('risk_neutral'), 150)} | **{r.get('pm_rating')}** | {fm_txt} |")

    # quant appendix
    L.append("\n---\n## Appendix A - Portfolio snapshot")
    L.append(f"Total {fmt_money(acct['total_value'])} | invested {fmt_money(acct['invested'])} | cash "
             f"{fmt_money(acct['cash'])} ({acct['cash_pct']}%) | unrealized P&L {fmt_money(acct['unrealized_pnl'])}\n")
    L.append("| Ticker | Shares | Avg cost | Price | Value | Weight | P&L % | 1m % | 3m % | vs SMA200 | Vol % | Beta 1y / 2y | Corr SPY 1y / 2y | Theme | Agent rating |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(stats["positions"], key=lambda r: -(r["weight_pct"] or 0)):
        L.append(f"| {r['ticker']} | {r['quantity']:g} | {r['avg_cost']:.2f} | {r['price'] or 0:.2f} | "
                 f"{fmt_money(r['market_value'])} | {r['weight_pct']:.1f}% | {r['unrealized_pnl_pct'] or 0:.1f} | "
                 f"{r.get('ret_1m_pct', '')} | {r.get('ret_3m_pct', '')} | {r.get('vs_sma200_pct', '')}% | {r.get('ann_vol_pct', '')} | "
                 f"{r.get('beta_1y', '')} / {r.get('beta_2y', '')}{' ⚠' if r.get('beta_unstable') else ''} | "
                 f"{r.get('corr_spy_1y', '')} / {r.get('corr_spy_2y', '')} | {r['theme']} | "
                 f"{reports.get(r['ticker'], {}).get('pm_rating', 'n/a')} |")
    rk = stats["risk"]
    L.append(f"\n**Risk:** portfolio vol {rk['portfolio_ann_vol_pct']}%/yr (SPY {rk['spy_ann_vol_pct']}%), beta "
             f"{rk['portfolio_beta_spy_1y']} (1y) / {rk['portfolio_beta_spy_2y']} (2y), 1y max drawdown at current weights "
             f"{rk['max_drawdown_1y_current_weights_pct']}%, 1-day 95% VaR {fmt_money(rk['var95_1d_usd'])}.")
    L.append("Risk contribution: " + ", ".join(f"{k} {v}%" for k, v in rk["risk_contribution_pct"].items()))
    L.append("\n**Stress** (beta-scaled uses the higher of 1y/2y beta): " + "; ".join(
        f"{k}: {fmt_money(v['pnl_usd'])} ({v['pnl_pct_of_account']}%)" + (f" [{v['window']}, SPY {v['spy_move_pct']}%]" if 'window' in v else "")
        for k, v in stats["stress"].items()))
    c = stats["concentration"]
    L.append(f"\n**Concentration:** effective positions {c['effective_positions']}, top 3 "
             + ", ".join(f"{t['ticker']} {t['weight_pct']}%" for t in c["top3"])
             + ". Buckets: " + ", ".join(f"{k} {v}%" for k, v in c["by_bucket_pct"].items()))
    L.append(f"\n**Correlation clusters (stocks linked by pairwise corr >= {stats['correlation_threshold']}, SPY excluded):** " + (
        "; ".join(f"{', '.join(g['tickers'])} (avg {g['avg_pairwise_corr']})" for g in stats["correlation_clusters"]) or "none"))
    def _fmt_regime(k, v):
        if isinstance(v, dict) and "value" in v:
            flag = " STALE" if v.get("stale") else ""
            yoy = f", YoY {v['yoy_pct']}%" if v.get("yoy_pct") is not None else ""
            return f"{k} {v['value']} ({v.get('date')}, age {v.get('age_days')}d{yoy}{flag})"
        return f"{k} {v}"
    L.append("\n**Market regime:** " + ", ".join(
        _fmt_regime(k, v) for k, v in stats["regime"].items() if k != "agent_ratings"))
    L.append("Agent ratings: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in stats["regime"].get("agent_ratings", {}).items()))
    if stats.get("data_checks"):
        L.append("\n**Data checks:**\n" + "\n".join(f"- {x}" for x in stats["data_checks"]))
    corr = pd.DataFrame(stats["correlation_matrix"])
    L.append("\n### Correlation matrix (1y daily returns)\n")
    L.append("| | " + " | ".join(corr.columns) + " |")
    L.append("|---" * (len(corr.columns) + 1) + "|")
    for idx, row in corr.iterrows():
        L.append(f"| {idx} | " + " | ".join(f"{v:.2f}" for v in row.values) + " |")
    if stats.get("target_drift"):
        L.append("\n### Drift vs guidance targets\n| Ticker | Current | Target | Drift | Shares to target |\n|---|---|---|---|---|")
        for x in stats["target_drift"]:
            L.append(f"| {x['ticker']} | {x['current_pct']}% | {x['target_pct']}{'%' if isinstance(x['target_pct'], (int, float)) else ''} | "
                     f"{x['drift_pct']:+}% | {x.get('shares_to_target', '')} |")
    if stats.get("watchlist"):
        L.append("\n### Watchlist candidates (not held)")
        L.append(f"| Ticker | Sector | Type | Price | 1m % | 3m % | 1y % | vs SMA200 | Vol % | Beta 1y | Corr to book | Avg corr to holdings | Vol change +3% from cash (pp) | Vol change swap 3% out of {stats.get('watchlist_swap_source')} (pp) | Agent rating |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for w in stats["watchlist"]:
            swap = next((v for k, v in w.items() if k.startswith("port_vol_change_pp_swap")), "")
            L.append(f"| {w['ticker']} | {w.get('sector') or ''} | {w.get('type') or ''} | {w.get('price') or 0:.2f} | "
                     f"{w.get('ret_1m_pct', '')} | {w.get('ret_3m_pct', '')} | {w.get('ret_1y_pct', '')} | {w.get('vs_sma200_pct', '')} | "
                     f"{w.get('ann_vol_pct', '')} | {w.get('beta_1y', '')} | {w.get('corr_to_book_1y', '')} | "
                     f"{w.get('avg_corr_to_holdings_1y', '')} | {w.get('port_vol_change_pp_add3pct_from_cash', '')} | {swap} | "
                     f"{w.get('agent_rating') or 'no report'} |")
    if macro:
        L.append("\n### Macro (FRED)")
        for k, v in macro.items():
            if "value" in v:
                extra = f", 1y ago {v['value_1y_ago']}" if "value_1y_ago" in v else ""
                extra += f", YoY {v['yoy_pct']}%" if "yoy_pct" in v else ""
                extra += f", age {v['age_days']}d" if "age_days" in v else ""
                extra += " STALE" if v.get("stale") else ""
                L.append(f"- {k}: {v['value']} ({v['date']}{extra})")
    L.append("\n## Appendix B - Source agent reports")
    for tk, r in reports.items():
        L.append(f"- {tk}: {r['pm_rating']} (trader {r['trader_action']}), analysis date {r['analysis_date']}, `logs/reports/{r['report_dir']}`")
    if calls:
        tot_cost = sum((c.get("usage") or {}).get("cost") or 0 for c in calls)
        tot_s = sum(c.get("seconds") or 0 for c in calls)
        L.append("\n## Appendix C - LLM calls")
        L.append("| Step | Model | Mode | Seconds | Tokens in / out | Cost (USD) |\n|---|---|---|---|---|---|")
        for c in calls:
            u = c.get("usage") or {}
            L.append(f"| {c['label']} | {c.get('model_req', c.get('model'))} | {c['mode']} | {c['seconds']} | "
                     f"{u.get('prompt_tokens')} / {u.get('completion_tokens')} | {u.get('cost', 'n/a')} |")
        L.append(f"| **Total** | | | {tot_s:.0f} | | {tot_cost:.4f} |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="TradingAgents fund manager add-on")
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="decision date (output folder)")
    ap.add_argument("--book", default=str(DATA_ROOT / "my_book.json"))
    ap.add_argument("--targets", default=str(DATA_ROOT / "targets.json"))
    ap.add_argument("--watchlist", default=str(DATA_ROOT / "watchlist.json"))
    ap.add_argument("--no-watchlist", action="store_true")
    ap.add_argument("--themes", default=str(ADDON_DIR / "themes.json"))
    ap.add_argument("--reports-dir", default=str(DATA_ROOT / "logs" / "reports"))
    ap.add_argument("--out-dir", default=str(DATA_ROOT / "fund_manager"))
    ap.add_argument("--since", default=None, help="only use reports generated on/after YYYY-MM-DD (UTC)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="fund manager model (draft + final)")
    ap.add_argument("--aggressive-model", default=os.getenv("FM_AGGRESSIVE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--neutral-model", default=os.getenv("FM_NEUTRAL_MODEL", QUICK_MODEL),
                    help="must differ from the fund manager; default is the Luna quick model")
    ap.add_argument("--conservative-model", default=os.getenv("FM_CONSERVATIVE_MODEL", QUICK_MODEL),
                    help="default is Luna. Opus is not the default (cost); it needs --allow-opus")
    ap.add_argument("--allow-opus", action="store_true",
                    help="permit an Opus model if one was passed explicitly")
    ap.add_argument("--risk-rounds", type=int, default=1, help="risk committee rounds (0 = no debate)")
    ap.add_argument("--reasoning-effort", default=os.getenv("FUND_MANAGER_REASONING_EFFORT", "high"),
                    help="fund manager effort: low|medium|high|none")
    ap.add_argument("--analyst-effort", default="medium", help="risk analysts effort: low|medium|high|none")
    ap.add_argument("--cash-min", type=float, default=None)
    ap.add_argument("--cash-max", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--analyst-max-tokens", type=int, default=12000)
    ap.add_argument("--max-report-chars", type=int, default=6000, help="per-section cap for agent text")
    ap.add_argument("--include-risk", action="store_true", help="also feed the per-stock risk-debate notes")
    ap.add_argument("--corr-threshold", type=float, default=0.5)
    ap.add_argument("--no-macro", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="quant snapshot only, no model call")
    args = ap.parse_args()

    book = load_json(Path(args.book))
    if not book:
        log(f"book not found: {args.book}")
        return 2
    targets = load_json(Path(args.targets)) or {}
    themes = load_json(Path(args.themes), {"themes": {}, "buckets": {}})
    tickers = [p["ticker"] for p in book["positions"]]
    cp = targets.get("cash_policy") or {}
    cash_policy = {"min_pct": args.cash_min if args.cash_min is not None else cp.get("min_pct", 5),
                   "max_pct": args.cash_max if args.cash_max is not None else cp.get("max_pct", 35),
                   "soft_default_pct": cp.get("soft_default_pct", targets.get("weights_pct", {}).get("cash", 12)),
                   "note": "Dynamic: the fund manager chooses the cash target each run inside this band."}
    watchlist = [] if args.no_watchlist else load_watchlist(Path(args.watchlist), tickers)
    wl_tk = [c["ticker"] for c in watchlist]

    found = find_latest_reports(Path(args.reports_dir), tickers, args.since)
    # candidates always use their latest existing report, however old (flagged with its analysis date)
    found.update(find_latest_reports(Path(args.reports_dir), wl_tk, None))
    missing = [t for t in tickers if t not in found]
    log(f"holdings: reports for {len(tickers) - len(missing)}/{len(tickers)}" + (f"; missing: {', '.join(missing)}" if missing else ""))
    if watchlist:
        have = [t for t in wl_tk if t in found]
        log(f"watchlist: {len(watchlist)} candidates, {len(have)} with agent reports" + (f" ({', '.join(have)})" if have else ""))
    reports = {tk: parse_report(tk, found[tk], args.include_risk, args.max_report_chars) for tk in tickers + wl_tk if tk in found}

    log("downloading ~2y daily prices from yfinance ...")
    close = fetch_prices_with_regime(tickers + wl_tk)
    quotes = live_quotes(tickers + wl_tk, close)
    meta = ticker_meta(tickers)
    macro = None
    if not args.no_macro:
        log("FRED macro snapshot ...")
        macro = fred_snapshot(datetime.now().strftime("%Y-%m-%d"))
    stats = build_stats(book, close, quotes, themes, targets, meta, args.corr_threshold, watchlist, reports, macro, cash_policy)
    acct = stats["account"]
    log(f"account ${acct['total_value']:,.0f} (cash {acct['cash_pct']}%), vol {stats['risk']['portfolio_ann_vol_pct']}%, "
        f"beta {stats['risk']['portfolio_beta_spy_1y']} (1y) / {stats['risk']['portfolio_beta_spy_2y']} (2y)")
    for msg in stats["data_checks"]:
        log(f"  data check: {msg}")

    run_id = datetime.now().strftime("%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out_dir = Path(args.out_dir) / args.date / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = "snapshot_" if args.no_llm else ""
    context = build_context(book, targets, stats, reports, macro, args.date)
    (out_dir / f"{pre}prompt.md").write_text(FM_SYSTEM + "\n\n" + context, encoding="utf-8")
    (out_dir / f"{pre}portfolio_stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

    args.model = _without_opus(args.model, args.allow_opus, "fund manager")
    args.aggressive_model = _without_opus(args.aggressive_model, args.allow_opus, "aggressive")
    args.conservative_model = _without_opus(args.conservative_model, args.allow_opus, "conservative")
    args.neutral_model = _without_opus(args.neutral_model, args.allow_opus, "neutral")
    if args.neutral_model == args.model:
        replacement = QUICK_MODEL if args.model != QUICK_MODEL else DEFAULT_MODEL
        if replacement == args.model:
            replacement = QUICK_MODEL
        log(f"neutral model {args.neutral_model} matches the fund manager; using {replacement} instead")
        args.neutral_model = replacement
    models = {"fm": args.model, "aggressive": args.aggressive_model, "conservative": args.conservative_model,
              "neutral": args.neutral_model,
              "note": "Conservative and neutral default to openai/gpt-6-luna (the quick model already used "
                      "for per-stock runs). Opus is not a default. Neutral is kept different from the fund manager."}
    eff = lambda e: None if e == "none" else e  # noqa: E731
    calls, draft, debate, final, check_d, check_f = [], None, [], None, None, None
    if not args.no_llm:
        log(f"[1/3] fund manager DRAFT ({args.model}, prompt ~{len(context) // 4:,} tokens) ...")
        r = call_llm(args.model, FM_SYSTEM, context + "\n\n## Task\nWrite your DRAFT whole-portfolio plan. Every holding must "
                     "appear exactly once in `trades` (HOLD 0 if unchanged); add watchlist buys only if justified.",
                     DRAFT_SCHEMA, "fund_manager_draft", args.max_tokens, eff(args.reasoning_effort), "fm/draft")
        r["model_req"] = args.model
        calls.append(r)
        draft = r["parsed"]
        if draft:
            check_d = validate_trades(draft, stats)
        if draft and args.risk_rounds > 0:
            log(f"[2/3] risk committee debate, {args.risk_rounds} round(s) ...")
            debate = run_debate(context, {**draft, "_post_trade_check": check_d}, models, args.risk_rounds,
                                args.analyst_max_tokens, eff(args.analyst_effort), args.date)
            calls += [{"label": f"risk/{t['role']} r{t['round']}", "model_req": t["model"], "mode": t["mode"],
                       "seconds": t["seconds"], "usage": t["usage"]} for t in debate]
            log("[3/3] fund manager FINAL ...")
            deb_txt = "\n\n".join(f"[Round {t['round']}] {t['role'].upper()} ({t['model']}):\n"
                                  + json.dumps(t["parsed"] or t["content"], indent=1) for t in debate)
            user = (context + "\n\n## Your DRAFT plan\n" + json.dumps(draft, indent=1)
                    + "\n\n## Post-trade check of the draft (computed)\n" + json.dumps(check_d, indent=1)
                    + "\n\n## Portfolio risk committee debate\n" + deb_txt
                    + "\n\n## Task\nIssue the FINAL whole-portfolio plan. Weigh each analyst's points on their merits "
                      "(do not just average), revisit the cash target, and list every change vs the draft with the reason "
                      "in `changes_vs_draft` (empty list if none). Summarise how you treated the key debate points in "
                      "`debate_takeaways`. Every holding must appear exactly once in `trades`.")
            r = call_llm(args.model, FM_SYSTEM, user, FINAL_SCHEMA, "fund_manager_final", args.max_tokens,
                         eff(args.reasoning_effort), "fm/final")
            r["model_req"] = args.model
            calls.append(r)
            final = r["parsed"]
            if final:
                check_f = validate_trades(final, stats)

    md = render_markdown(args.date, models, stats, reports, macro, calls, draft, debate, final, check_d, check_f, targets)
    stem = "portfolio_snapshot" if args.no_llm else "portfolio_report"
    (out_dir / f"{stem}.md").write_text(md, encoding="utf-8")
    payload = {"date": args.date, "run_id": run_id, "models": models,
               "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "reports": {tk: {k: v for k, v in r.items() if k not in ("pm_decision", "research_manager", "risk_debate")}
                           for tk, r in reports.items()},
               "stats": stats, "macro": macro, "draft": draft, "draft_check": check_d, "debate": debate,
               "final": final, "final_check": check_f, "diff_vs_draft": diff_plans(draft, final),
               "llm_calls": [{k: v for k, v in c.items() if k not in ("content", "parsed")} for c in calls],
               "raw_text_fallbacks": [c["content"] for c in calls if c.get("mode") == "text"]}
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    latest = {"date": args.date, "run_id": run_id, "path": f"{args.date}/{run_id}",
              "kind": "snapshot" if args.no_llm else "report",
              "generated_utc": payload["generated_utc"]}
    (Path(args.out_dir) / "latest.json").write_text(json.dumps(latest, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"FUND MANAGER - {args.date}   account ${acct['total_value']:,.0f}")
    plan, chk = (final, check_f) if final else (draft, check_d)
    if plan and chk:
        print(("FINAL: " if final else "DRAFT: ") + plan.get("headline", ""))
        cd = plan.get("cash_decision") or {}
        print(f"  target cash {cd.get('target_cash_pct')}% ({cd.get('regime')}), post-trade cash ~${chk['ending_cash']:,.0f} ({chk['ending_cash_pct']}%)")
        order = {tk: i for i, tk in enumerate(plan.get("priority_order", []))}
        for t in sorted(chk["trades"], key=lambda t: order.get(t["ticker"], 99)):
            if t["action"] != "HOLD" and t["shares"]:
                print(f"  {t['action']:4s} {t['shares']:>4d} {t['ticker']:6s} ~${t['value_usd']:>9,.0f}  ({t.get('decision', '')})")
        for d in diff_plans(draft, final):
            print(f"  changed vs draft: {d['ticker']}: {d['draft']} -> {d['final']}")
        risk = chk.get("post_trade_risk") or {}
        if risk:
            print(f"  post-trade beta {risk.get('beta_spy')} | AI look-through {risk.get('ai_lookthrough_pct')}% | top 3 {risk.get('top3_pct')}%")
        for w in chk["warnings"]:
            print(f"  ! {w}")
    if calls:
        print(f"  LLM calls {len(calls)}, ~{sum(c.get('seconds') or 0 for c in calls) / 60:.1f} min, "
              f"cost ~${sum((c.get('usage') or {}).get('cost') or 0 for c in calls):.2f}")
    print(f"report: {out_dir / (stem + '.md')}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
