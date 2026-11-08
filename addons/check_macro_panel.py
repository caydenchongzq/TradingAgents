"""Offline checks for the shared point-in-time panel. No network and no model call."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import macro_panel as mp


class _Fred:
    def __init__(self):
        self.calls = []

    def _fred_today(self):
        return "2026-11-07"

    def _request(self, path, params):
        self.calls.append((path, dict(params)))
        if params["series_id"] == "CPILFESL":
            return {"observations": [
                {"date": "2026-10-01", "value": "110"},
                {"date": "2025-09-01", "value": "100"},
            ]}
        if params["series_id"] == "DGS10":
            return {"observations": [{"date": "2026-11-06", "value": "4.1"}]}
        return {"observations": []}


def main() -> int:
    params = mp.fred_observation_params("2026-11-06", "DGS10", "2024-08-28", "2026-11-07")
    assert params["observation_end"] == "2026-11-06"
    assert params["realtime_start"] == "2026-11-06"
    assert params["realtime_end"] == "2026-11-06"
    future = mp.fred_observation_params("2026-11-08", "DGS10", "2024-08-28", "2026-11-07")
    assert future["realtime_start"] == "2026-11-07"
    assert future["observation_end"] == "2026-11-08"

    fred = _Fred()
    snap = mp.fred_snapshot("2026-11-06", {"10y_treasury": "DGS10", "core_cpi_index": "CPILFESL"}, fred=fred)
    assert snap["as_of"] == "2026-11-06" and snap["vintage"] == "2026-11-06"
    assert all("realtime_start" in call[1] and "realtime_end" in call[1] for call in fred.calls)
    ten = snap["10y_treasury"]
    assert ten["value"] == 4.1 and ten["as_of"] == "2026-11-06" and ten["stale"] is False
    cpi = snap["core_cpi_index"]
    assert cpi["as_of"] == "2026-11-06" and cpi["stale"] is False
    assert cpi["yoy_pct"] == 10.0

    stale = mp.series_from_observations("vix", "VIXCLS", [{"date": "2026-10-01", "value": "20"}], "2026-11-06", "2026-11-06")
    assert stale["stale"] is True and stale["as_of"] == "2026-11-06"

    idx = pd.bdate_range("2026-10-01", "2026-11-10")
    frame = pd.DataFrame({"SPY": range(len(idx))}, index=idx)
    clipped = mp.clip_frame(frame, "2026-11-06")
    assert clipped.index.max() <= pd.Timestamp("2026-11-06")
    assert len(clipped) < len(frame)

    rows = mp.filter_headlines([
        {"source": "yfinance", "title": "later", "date": "2026-11-08T15:00:00Z"},
        {"source": "alpha_vantage", "title": "ok", "date": "20261106T120000"},
        {"source": "yfinance", "title": "undated"},
    ], "2026-11-06")
    titles = [row["title"] for row in rows]
    assert titles == ["ok", "undated"], titles
    assert rows[0]["headline_date"] == "2026-11-06" and rows[0]["as_of"] == "2026-11-06"
    assert rows[1]["timing"] == "undated"
    live = mp.mark_live({"earnings_days": 3}, "earnings_days")
    assert live["timing"] == "live" and live["live_fields"] == ["earnings_days"]
    start, end = mp.history_window("2026-11-06", 430)
    assert end == "2026-11-07" and start == "2025-09-02", (start, end)

    # theme scout and fund manager download with start/end that stop at the decision date
    import types
    seen = []

    def fake_download(symbols, **kw):
        seen.append(kw)
        idx = pd.bdate_range("2026-10-01", "2026-11-10")
        cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close"], list(symbols)])
        return pd.DataFrame(1.0, index=idx, columns=cols)

    sys.modules["yfinance"] = types.SimpleNamespace(download=fake_download)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "theme_scout"))
    import theme_scout as ts
    panels = ts.download_ohlc(["SPY", "AAA"], "2026-11-06")
    assert seen[-1].get("end") == "2026-11-07" and "period" not in seen[-1], seen[-1]
    assert panels["Close"].index.max() <= pd.Timestamp("2026-11-06")
    ts.download_ohlc(["SPY"])
    assert seen[-1].get("period") == "14mo"
    sys.path.insert(0, str(Path(__file__).resolve().parent / "fund_manager"))
    import fund_manager as fm
    close = fm.fetch_prices(["AAA"], as_of="2026-11-06")
    assert seen[-1].get("end") == "2026-11-07" and seen[-1].get("start") == "2024-10-27", seen[-1]
    assert close.index.max() <= pd.Timestamp("2026-11-06")
    assert fm._period_days("14mo") == 439 and fm._period_days("2y") == 740
    del sys.modules["yfinance"]
    print("check_macro_panel: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
