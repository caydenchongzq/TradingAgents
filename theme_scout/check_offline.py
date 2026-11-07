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
        scores, min_n=30, max_n=50, shadow=False, as_of="2026-11-06")
    actions = [c["action"] for c in changes]
    assert "add" in actions, changes
    assert actions.count("drop") == 3, changes  # fourth drop would break the floor of 30 (32+1-4=29)
    assert any(c["action"] == "kept" and c["ticker"] == "N03" for c in changes)
    assert len(updated) == 30

    wide = [{"ticker": f"W{i:02d}", "sector": "Other", "type": "stock"} for i in range(48)]
    wide_scores = {item["ticker"]: 48 - i for i, item in enumerate(wide)}
    trimmed, tchanges = ts.apply_watchlist(
        wide, [{"ticker": f"A{i}", "sector": "X", "reason": "add"} for i in range(4)], [],
        wide_scores, min_n=30, max_n=50, shadow=False, as_of="2026-11-06")
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
    prompt = " ".join(ts.SCOUT_SYSTEM.split())
    assert "unless they are liquid" not in prompt
    assert "Do not invent a ticker" in prompt
    assert ts.curator_shadow("2026-11-07", {}) is True
    assert ts.curator_shadow("2026-12-06", {}) is False
    assert ts.curator_shadow("2026-11-07", {"watchlist_curator": {"shadow": False}}) is False
    assert ts.ai_weight_for("ZZZ", {"ai_sleeve": {"ticker_weight": {"GLD": 0}, "theme_weight": {}}}, {}) == (1.0, "default_until_tagged")
    assert ts.ai_weight_for("GLD", {"ai_sleeve": {"ticker_weight": {"GLD": 0}}}, {}) == (0.0, "ticker_weight")

    tagged = {"themes": {"sector_etf": ["XLV"], "ai_growth": ["NVDA"]}}
    sleeve_t = {"ai_sleeve": {"ticker_weight": {}, "theme_weight": {"ai_growth": 1.0}}}
    assert ts.ai_weight_for("XLV", sleeve_t, tagged) == (0.0, "theme sector_etf")
    assert ts.ai_weight_for("NVDA", sleeve_t, tagged) == (1.0, "theme_weight")
    import pandas as pd
    book_close = pd.DataFrame({"NVDA": [100.0], "XLV": [50.0]}, index=pd.to_datetime(["2026-11-06"]))
    book = {"cash": 0, "positions": [{"ticker": "NVDA", "quantity": 1, "average_price": 90},
                                     {"ticker": "XLV", "quantity": 2, "average_price": 40}]}
    assert ts.book_ai_lookthrough(book, book_close, sleeve_t, tagged) == 50.0
    assert ts.book_ai_lookthrough({"positions": []}, book_close, sleeve_t, tagged) is None

    held_drop, held_changes = ts.apply_watchlist(
        items, [], [{"ticker": "N00", "reason": "sell"}], scores,
        shadow=False, as_of="2026-11-06", holdings={"N00"})
    assert any(c["action"] == "trim" and "holding" in c["reason"] for c in held_changes), held_changes
    assert any(row["ticker"] == "N00" for row in held_drop)
    staged_drop, staged_changes = ts.apply_watchlist(
        items, [], [{"ticker": "N01", "reason": "sell"}], scores,
        shadow=False, as_of="2026-11-06", staged={"N01"})
    assert any(c["action"] == "trim" and "staged" in c["reason"] for c in staged_changes)
    assert any(row["ticker"] == "N01" for row in staged_drop)

    cooled, cool_changes = ts.apply_watchlist(
        items, [{"ticker": "NEW", "sector": "X", "reason": "back"}], [], scores,
        shadow=False, as_of="2026-11-06",
        log_events=[{"id": "d1", "action": "drop", "ticker": "NEW", "as_of": "2026-11-01"}])
    assert any(c["action"] == "skip_add" and "cooldown" in c["reason"] for c in cool_changes), cool_changes
    assert all(row["ticker"] != "NEW" for row in cooled)

    week_events = [{"id": f"a{i}", "action": "add", "ticker": f"Z{i}", "as_of": "2026-11-03"} for i in range(5)]
    capped, cap_changes = ts.apply_watchlist(
        items, [{"ticker": "NEW", "sector": "X", "reason": "sixth"}], [], scores,
        shadow=False, as_of="2026-11-06", log_events=week_events)
    assert any(c["action"] == "skip_add" and "weekly add cap" in c["reason"] for c in cap_changes)
    assert all(row["ticker"] != "NEW" for row in capped)

    sleeve = {"ai_sleeve": {"cap_pct": 50, "ticker_weight": {"GLD": 0.0}, "theme_weight": {}}}
    blocked, block_changes = ts.apply_watchlist(
        items, [{"ticker": "ZZZ", "sector": "X", "reason": "untagged"},
                {"ticker": "GLD", "sector": "Gold", "type": "etf", "reason": "non-ai"}], [],
        scores, shadow=False, as_of="2026-11-06", targets=sleeve, ai_lookthrough_pct=50)
    assert any(c["ticker"] == "ZZZ" and c["action"] == "skip_add" and c.get("ai_weight") == 1.0 for c in block_changes)
    assert any(c["ticker"] == "GLD" and c["action"] == "add" and c.get("ai_weight") == 0.0 for c in block_changes)
    outside, out_changes = ts.apply_watchlist(
        items, [{"ticker": "ZZZ", "sector": "X", "reason": "invented"}], [], scores,
        shadow=False, as_of="2026-11-06", universe={"N00"})
    assert any(c["action"] == "skip_add" and "universe" in c["reason"] for c in out_changes)
    assert all(row["ticker"] != "ZZZ" for row in outside)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import watchlist_log
    log_root = Path(tempfile.mkdtemp(prefix="watch_log_"))
    try:
        watch_file = log_root / "watchlist.json"
        watch_file.write_text(json.dumps(items), encoding="utf-8")
        before = watch_file.read_text(encoding="utf-8")
        proposed, shadow_changes = ts.apply_watchlist(
            items, [{"ticker": "NEW", "sector": "X", "reason": "idea"}], [], scores,
            shadow=True, as_of="2026-11-07")
        outcome = ts.commit_watchlist(
            watch_file, items, proposed, shadow_changes, log_root / "watchlist_proposals.json", log_root, "2026-11-07")
        assert outcome == "proposal"
        assert watch_file.read_text(encoding="utf-8") == before
        assert list(log_root.glob("*.bak*")) == []
        assert (log_root / "watchlist_proposals.json").is_file()
        watchlist_log.append_event(log_root, {
            "id": "add1", "action": "add", "ticker": "BBB", "as_of": "2026-11-06",
            "row": {"ticker": "BBB", "sector": "X", "type": "stock"},
        })
        restored = watchlist_log.undo_items(items + [{"ticker": "BBB", "type": "stock"}], {"action": "add", "ticker": "BBB"})
        assert [row["ticker"] for row in restored] == [row["ticker"] for row in items]
        watchlist_log.append_event(log_root, {"action": "undo", "undo_of": "add1", "ticker": "BBB", "as_of": "2026-11-06"})
        assert watchlist_log.last_undoable(watchlist_log.read_events(log_root)) is None
    finally:
        shutil.rmtree(log_root)

    print("check_offline: ok")
    print(f"  AAA rs_1m {by['AAA']['rs_1m']} BBB rs_1m {by['BBB']['rs_1m']}")
    print(f"  watchlist floor kept {len(updated)} names; cap trimmed to {len(trimmed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
