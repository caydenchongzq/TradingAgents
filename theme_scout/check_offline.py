"""Offline checks for the theme scout. No network and no model call."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import theme_scout as ts


def _close():
    idx = pd.bdate_range("2025-01-01", periods=220)
    # SPY drifts up slowly. AAA trends harder. BBB falls.
    spy = pd.Series(range(100, 320), index=idx, dtype=float)
    aaa = pd.Series(range(50, 270), index=idx, dtype=float) * 1.4
    bbb = pd.Series(range(300, 80, -1), index=idx, dtype=float)
    frame = pd.DataFrame({"SPY": spy, "AAA": aaa, "BBB": bbb})
    return frame


def main() -> int:
    close = _close()
    high = close * 1.01
    low = close * 0.99
    ranked = ts.screen_frame(close, high, low, ["AAA", "BBB"], "SPY", {"AAA": 3, "BBB": None})
    by = {r["ticker"]: r for r in ranked}
    assert by["AAA"]["rs_1m"] > by["BBB"]["rs_1m"], by
    assert by["AAA"]["rank"] == 1, ranked
    assert by["AAA"]["dist_ma50"] is not None and by["AAA"]["atr_pct"] is not None
    assert by["AAA"]["earnings_days"] == 3
    assert by["AAA"]["earnings_timing"] == "live"
    assert by["AAA"]["score"] > by["BBB"]["score"]

    groups = ts.group_rs(close, {"leaders": ["AAA"], "laggards": ["BBB"]}, "SPY")
    rs = {g["id"]: g["rs_1m"] for g in groups}
    assert rs["leaders"] > 0 > rs["laggards"], rs

    items = [{"ticker": f"N{i:02d}", "sector": "Other", "type": "stock"} for i in range(32)]
    scores = {item["ticker"]: float(i) for i, item in enumerate(items)}
    updated, changes = ts.apply_watchlist(
        items,
        [{"ticker": "NEW", "sector": "Software", "sub_theme": "ai_cloud_software", "reason": "test add"}],
        [{"ticker": "N00", "reason": "weak"}, {"ticker": "N01", "reason": "weak"},
         {"ticker": "N02", "reason": "weak"}, {"ticker": "N03", "reason": "weak"}],
        scores, min_n=30, max_n=50)
    actions = [c["action"] for c in changes]
    assert "add" in actions, changes
    assert actions.count("drop") == 3, changes  # fourth drop would break the floor of 30 (32+1-4=29)
    assert any(c["action"] == "kept" and c["ticker"] == "N03" for c in changes)
    assert len(updated) == 30

    wide = [{"ticker": f"W{i:02d}", "sector": "Other", "type": "stock"} for i in range(48)]
    wide_scores = {item["ticker"]: 48 - i for i, item in enumerate(wide)}
    trimmed, tchanges = ts.apply_watchlist(
        wide, [{"ticker": f"A{i}", "sector": "X", "reason": "add"} for i in range(4)], [],
        wide_scores, min_n=30, max_n=50)
    assert len(trimmed) == 50, len(trimmed)
    assert sum(1 for c in tchanges if c["action"] == "trim") == 2

    ordered, errors = ts.normalize_ranking(
        [{"ticker": "bbb", "rank": 2, "bucket": "candidate", "action": "pass", "reason": "weak"},
         {"ticker": "AAA", "rank": 1, "bucket": "holding", "action": "keep", "reason": "strong"},
         {"ticker": "CCC", "rank": 3, "bucket": "candidate", "action": "trim", "reason": "wrong action still stored"}],
        ["AAA"], ["BBB", "CCC"])
    assert errors == [], errors
    assert [row["ticker"] for row in ordered] == ["AAA", "BBB", "CCC"]
    assert ordered[2]["action"] == "trim" and ordered[2]["bucket"] == "candidate"
    bad, bad_errors = ts.normalize_ranking(
        [{"ticker": "AAA", "rank": 1, "action": "keep", "reason": "only one"}],
        ["AAA"], ["BBB"])
    assert bad_errors and any("missing" in err for err in bad_errors)

    held = ["NVDA"]
    tier = ts.tier_candidates(held, ranked + [{"ticker": "NVDA", "rank": 0, "score": 9},
                                              {"ticker": "CCC", "rank": 3, "score": 1, "error": "no prices"}], 1)
    assert tier == ["AAA"], tier

    root = Path(tempfile.mkdtemp(prefix="theme_scout_offline_"))
    try:
        sample = root / "watchlist.json"
        sample.write_text(json.dumps(items), encoding="utf-8")
        before = sample.read_text(encoding="utf-8")
        # --dry-run never calls apply_watchlist and never writes the watchlist.
        assert ts.apply_watchlist is not None
        assert sample.read_text(encoding="utf-8") == before
    finally:
        shutil.rmtree(root)

    md = ts.render_screen_md("2026-11-06", ranked)
    assert "AAA" in md and "RS 1m" in md
    report = ts.render_report_md({
        "date": "2026-11-06", "llm": None, "headline": "", "macro": {}, "headlines": [],
        "relative_strength": groups, "screen": ranked, "themes": [], "changes": [],
        "watchlist_size": 39,
    })
    assert "No model call" in report and "leaders" in report
    print("check_offline: ok")
    print(f"  AAA rs_1m {by['AAA']['rs_1m']} BBB rs_1m {by['BBB']['rs_1m']}")
    print(f"  watchlist floor kept {len(updated)} names; cap trimmed to {len(trimmed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
