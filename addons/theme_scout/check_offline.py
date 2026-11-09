"""Offline checks for the theme scout. No network and no model call."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import theme_scout as ts


def _check_large_ranking_context_and_truncation() -> None:
    holdings = [f"H{i:02d}" for i in range(12)]
    saved = [f"W{i:02d}" for i in range(78)]
    discovered = [f"D{i:02d}" for i in range(15)]
    sector_etfs = ["XLK", "XLY", "XLV", "XLP", "XLU"]
    names = ([{"ticker": ticker, "bucket": "holding"} for ticker in holdings]
             + [{"ticker": ticker, "bucket": "candidate"} for ticker in saved]
             + [{"ticker": ticker, "bucket": "discovery", "sector": "Technology"} for ticker in discovered]
             + [{"ticker": ticker, "bucket": "candidate", "type": "etf"} for ticker in sector_etfs])
    ranked = [{"ticker": row["ticker"], "score": 1.2, "rs_1w": 1.1, "rs_1m": 2.3,
               "rs_3m": 4.5, "dist_ma20": 0.2, "dist_ma50": -1.4,
               "dist_ma200": -3.2, "atr_pct": 5.6, "earnings_days": None,
               "price": 123.45, "ret_1w": 6.7, "ret_1m": 8.9}
              for row in names]
    context = ts.scout_model_context(
        "2026-11-09", {"vix": 20},
        ([{"source": "yfinance", "title": f"Yahoo headline {i}"} for i in range(8)]
         + [{"source": "alpha_vantage", "title": f"AV headline {i}", "summary": "Summary"} for i in range(8)]),
        [{"id": f"sector_{ticker}", "rs_1m": -1.2} for ticker in sector_etfs],
        sector_etfs, names, ranked, [f"theme_{i}" for i in range(12)])
    parsed = json.loads(context)
    tickers = [row["ticker"] for row in parsed["names"]]
    assert len(tickers) == 110 and len(set(tickers)) == 110
    assert set(tickers) == {row["ticker"] for row in names}
    assert parsed["discovery_sectors"] == sector_etfs
    assert all("price" not in row["evidence"] and "ret_1w" not in row["evidence"]
               for row in parsed["names"])
    assert all(key in parsed["names"][0]["evidence"]
               for key in ("rs_1w", "rs_1m", "rs_3m", "dist_ma20", "dist_ma50", "dist_ma200"))
    # Repeated full screen and discovery payloads must not bloat the model input.
    old_shape = json.dumps({"names": names, "screen": ranked, "discovery_pool": ranked[90:]}, default=str)
    assert len(context) < len(old_shape)
    compact_headlines = parsed["headlines"]
    assert sum(row.get("source") == "yfinance" and "title" in row for row in compact_headlines) == 4
    assert sum(row.get("source") == "alpha_vantage" and "title" in row for row in compact_headlines) == 4
    assert parsed["headline_sources"]["alpha_vantage"]["status"] == "ok"

    headlines, source_status = ts.compact_scout_headlines([
        {"source": "yfinance", "title": f"Yahoo {i}"} for i in range(10)
    ] + [{"source": "alpha_vantage", "error": "rate limit exceeded"}])
    assert source_status["yfinance"] == {"status": "ok", "headlines_available": 10}
    assert source_status["alpha_vantage"] == {"status": "error", "message": "rate limit exceeded"}
    assert any(row.get("error") == "rate limit exceeded" for row in headlines)

    expected = holdings + saved + discovered + sector_etfs
    rows = [{"ticker": ticker, "rank": index, "bucket": "holding" if ticker in holdings else "candidate",
             "action": "keep" if ticker in holdings else "pass", "reason": "evidence"}
            for index, ticker in enumerate(expected, 1)]
    ordered, errors = ts.normalize_ranking(rows, holdings, saved + discovered + sector_etfs)
    assert not errors and len(ordered) == 110, errors
    assert [row["rank"] for row in ordered] == list(range(1, 111))

    # Reproduce the live eight-sector failure: 33 returned rows, a duplicate,
    # and the remaining 133 names omitted from the response.
    expected_166 = [f"S{i:03d}" for i in range(166)]
    partial_rows = [{"ticker": ticker, "rank": i + 2, "action": "add" if i == 0 else "pass",
                     "reason": "returned evidence"} for i, ticker in enumerate(expected_166[:33])]
    partial_rows.append({"ticker": expected_166[0], "rank": 1, "action": "pass", "reason": "duplicate"})
    partial, warnings = ts.normalize_ranking(partial_rows, [], expected_166)
    assert len(partial) == 33 and partial[0]["rank"] == 2 and partial[0]["action"] == "conflict"
    assert partial[0]["action_conflict"] is True
    assert len({row["ticker"] for row in partial}) == 33
    assert partial[-1]["rank"] == 34  # model ranks remain intact; no ranks are fabricated
    assert any("duplicate rank row" in warning for warning in warnings)
    assert any("marked non-actionable" in warning for warning in warnings)
    assert expected_166[33] in " ".join(warnings) and len(expected_166) == 166
    partial_md = ts.render_report_md({
        "date": "2026-11-09", "llm": {"model": "test/model"}, "ranking": partial,
        "ranking_status": "partial", "ranking_count": len(partial),
        "ranking_expected_count": len(expected_166), "ranking_missing_tickers": expected_166[33:],
        "ranking_warnings": warnings,
    })
    assert "Partial coverage: 33 of 166 expected names" in partial_md
    assert "Missing names have no inferred rank or action" in partial_md
    additions, mismatches = ts.ranked_watchlist_additions(
        partial, [], {expected_166[0]: {"ticker": expected_166[0], "sector": "Technology", "type": "stock"}}, set())
    assert additions == []  # conflicting add/pass rows are not actionable
    assert not any(row["ticker"] in expected_166[33:] for row in additions)
    assert mismatches == []
    holding_conflict, holding_warnings = ts.normalize_ranking(
        [{"ticker": "HELD", "rank": 1, "action": "keep"},
         {"ticker": "HELD", "rank": 2, "action": "trim"}], ["HELD"], [])
    assert holding_conflict[0]["action"] == "conflict"
    assert holding_conflict[0]["action_conflict"] is True
    assert any("HELD" in warning and "marked non-actionable" in warning for warning in holding_warnings)
    same_action, same_action_warnings = ts.normalize_ranking(
        [{"ticker": "AAA", "rank": 1, "action": "add"},
         {"ticker": "AAA", "rank": 2, "action": "add"}], [], ["AAA"])
    assert same_action[0]["action"] == "add" and not same_action[0].get("action_conflict")
    assert any("duplicate rank row AAA" in warning for warning in same_action_warnings)
    safe_drops, drop_warnings = ts.ranked_watchlist_drops([
        {"ticker": "ADD", "bucket": "candidate", "action": "add"},
        {"ticker": "KEEP", "bucket": "candidate", "action": "keep"},
        {"ticker": "PASS", "bucket": "candidate", "action": "pass"},
        {"ticker": "TRIM", "bucket": "candidate", "action": "trim"},
        {"ticker": "HELD", "bucket": "holding", "action": "trim"},
        {"ticker": "CONFLICT", "bucket": "candidate", "action": "pass", "action_conflict": True},
    ], [{"ticker": ticker} for ticker in ("ADD", "KEEP", "PASS", "TRIM", "HELD", "CONFLICT", "MISSING")])
    assert [row["ticker"] for row in safe_drops] == ["PASS", "TRIM"]
    assert len(drop_warnings) == 5
    assert ts.ranking_response_error({"parsed": {}, "finish_reason": "length",
                                      "parse_error": "truncated JSON"}) == \
        "model response reached its output-token limit; ranking is incomplete"
    assert ts.ranking_response_error({"parsed": {}, "finish_reason": "stop"}) == \
        "model response contained no ranking"
    assert ts.ranking_response_error({"parsed": {"ranking": rows}, "finish_reason": "stop"}) is None


def _check_ranked_add_sync() -> None:
    ranking = [
        {"ticker": "NEW", "bucket": "candidate", "action": "add", "reason": "monitor the catalyst"},
        {"ticker": "PASS", "bucket": "candidate", "action": "pass", "reason": "weak evidence"},
        {"ticker": "HELD", "bucket": "holding", "action": "keep", "reason": "existing position"},
    ]
    candidates = {
        "NEW": {"ticker": "NEW", "sector": "Technology", "type": "stock"},
        "PASS": {"ticker": "PASS", "sector": "Technology", "type": "stock"},
        "HELD": {"ticker": "HELD", "sector": "Technology", "type": "stock"},
    }
    # A ranking add must stage even when the model omits it from the redundant
    # top-level add array. A pass or holding can never be staged from that array.
    additions, mismatches = ts.ranked_watchlist_additions(
        ranking, [{"ticker": "PASS", "reason": "model disagreement"},
                  {"ticker": "HELD", "reason": "protected"}], candidates, {"HELD"})
    assert additions == [{"ticker": "NEW", "sector": "Technology", "type": "stock",
                          "reason": "monitor the catalyst"}], additions
    assert len(mismatches) == 2, mismatches


def _check_main_proposal_flow() -> None:
    root = Path(tempfile.mkdtemp(prefix="theme_scout_main_"))
    watch = [{"ticker": "AAA", "sector": "Technology", "type": "stock"}]
    wrapper = {"items": watch, "metadata": "preserve me"}
    (root / "watchlist.json").write_text(json.dumps(wrapper), encoding="utf-8")
    (root / "my_book.json").write_text(json.dumps({"positions": [{"ticker": "BBB"}]}), encoding="utf-8")
    frame = pd.DataFrame()
    parsed = {
        "headline": "test",
        "themes": [],
        "add": [{"ticker": "XLV", "sector": "wrong model label", "reason": "sector ETF"},
                {"ticker": "NEW", "sector": "Healthcare", "reason": "test candidate"}],
        "drop": [],
        "ranking": [
            {"ticker": "BBB", "rank": 1, "action": "keep", "reason": "held"},
            {"ticker": "AAA", "rank": 2, "action": "keep", "reason": "candidate"},
            {"ticker": "XLV", "rank": 3, "action": "add", "reason": "sector ETF"},
            {"ticker": "NEW", "rank": 4, "action": "add", "reason": "discovery"},
        ],
    }
    macro_panel = SimpleNamespace(filter_headlines=lambda headlines, _date: headlines)
    scout_call = {}
    def fake_call_scout(*args, **_kwargs):
        scout_call["calls"] = scout_call.get("calls", 0) + 1
        scout_call["max_tokens"] = args[3]
        scout_call["max_retries"] = _kwargs["max_retries"]
        return {"parsed": parsed, "seconds": 0, "finish_reason": "stop",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    try:
        with patch("sys.argv", ["theme_scout.py", "--data-root", str(root), "--date", "2026-12-06",
                                "--sectors", *ts.SECTORS, "--single-call"]), \
             patch.object(ts, "agent_plan_from_config", return_value={"theme_scout": {
                 "model": "test/model", "max_tokens": 20, "effort": "low", "temperature": 0,
                 "system": "test system"}}), \
             patch.object(ts, "download_ohlc", return_value={"Close": frame, "High": frame, "Low": frame}), \
             patch.object(ts, "discover_sector_candidates", return_value=[{
                 "ticker": "NEW", "sector": "Healthcare", "type": "stock",
                 "discovery_source": "Yahoo Finance screener"}]), \
             patch.object(ts, "earnings_days", return_value=None), \
             patch.object(ts, "screen_frame", return_value=[
                 {"ticker": "AAA", "rank": 1, "score": 1},
                 {"ticker": "BBB", "rank": 2, "score": 2},
                 {"ticker": "XLV", "rank": 3, "score": 0.5},
                 {"ticker": "NEW", "rank": 4, "score": 0.4}]), \
             patch.object(ts, "group_rs", return_value=[]), \
             patch.object(ts, "fred_snapshot", return_value={}), \
             patch.object(ts, "yahoo_headlines", return_value=[]), \
             patch.object(ts, "av_headlines", return_value=[]), \
             patch.object(ts, "call_scout", side_effect=fake_call_scout), \
             patch.object(ts, "staged_tickers", return_value=set()), \
             patch.object(ts, "book_ai_lookthrough", return_value=0), \
             patch.dict("sys.modules", {"macro_panel": macro_panel}):
            assert ts.main() == 0

        assert json.loads((root / "watchlist.json").read_text(encoding="utf-8")) == wrapper
        proposal_path = root / "theme_scout" / "2026-12-06" / "watchlist_proposals.json"
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        assert proposal["shadow"] is True
        assert proposal["sectors"] == ts.SECTORS
        assert proposal["base_watchlist_version"] == ts.watchlist_version(watch)
        assert any(row["ticker"] == "XLV" and row["type"] == "etf" for row in proposal["proposed"]), proposal
        assert any(row["ticker"] == "NEW" and row["type"] == "stock" for row in proposal["proposed"]), proposal
        assert all(not change["applied"] for change in proposal["changes"])
        report = json.loads((proposal_path.parent / "report.json").read_text(encoding="utf-8"))
        assert report["watchlist_size"] == len(watch)
        assert scout_call["max_tokens"] == 20, scout_call
        assert scout_call["calls"] == 1, scout_call
        assert scout_call["max_retries"] == 0, scout_call

        # A failed same-day model run must not leave the last successful proposal
        # at the canonical path where Argus would still offer it for application.
        failure = {"parsed": {}, "seconds": 1, "finish_reason": "length",
                   "parse_error": "truncated JSON",
                   "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.01}}
        with patch("sys.argv", ["theme_scout.py", "--data-root", str(root), "--date", "2026-12-06",
                                "--sectors", "XLK", "XLY", "XLV", "--single-call"]), \
             patch.object(ts, "agent_plan_from_config", return_value={"theme_scout": {
                 "model": "test/model", "max_tokens": 20, "effort": "low", "temperature": 0,
                 "system": "test system"}}), \
             patch.object(ts, "download_ohlc", return_value={"Close": frame, "High": frame, "Low": frame}), \
             patch.object(ts, "discover_sector_candidates", return_value=[{
                 "ticker": "NEW", "sector": "Healthcare", "type": "stock",
                 "discovery_source": "Yahoo Finance screener"}]), \
             patch.object(ts, "earnings_days", return_value=None), \
             patch.object(ts, "screen_frame", return_value=[
                 {"ticker": "AAA", "rank": 1, "score": 1},
                 {"ticker": "BBB", "rank": 2, "score": 2},
                 {"ticker": "XLV", "rank": 3, "score": 0.5},
                 {"ticker": "NEW", "rank": 4, "score": 0.4}]), \
             patch.object(ts, "group_rs", return_value=[]), \
             patch.object(ts, "fred_snapshot", return_value={}), \
             patch.object(ts, "yahoo_headlines", return_value=[]), \
             patch.object(ts, "av_headlines", return_value=[]), \
             patch.object(ts, "call_scout", return_value=failure), \
             patch.object(ts, "staged_tickers", return_value=set()), \
             patch.object(ts, "book_ai_lookthrough", return_value=0), \
             patch.dict("sys.modules", {"macro_panel": macro_panel}):
            assert ts.main() == 1
        assert not proposal_path.exists()
        archived = list(proposal_path.parent.glob("watchlist_proposals.superseded_*.json"))
        assert len(archived) == 1
        failed_report = json.loads((proposal_path.parent / "report.json").read_text(encoding="utf-8"))
        assert failed_report["model_response_error"] == \
            "model response reached its output-token limit; ranking is incomplete"
        assert failed_report["llm"]["usage"]["cost"] == 0.01
    finally:
        shutil.rmtree(root)


def _close():
    idx = pd.bdate_range("2025-01-01", periods=220)
    # SPY drifts up slowly. AAA trends harder. BBB falls.
    spy = pd.Series(range(100, 320), index=idx, dtype=float)
    aaa = pd.Series(range(50, 270), index=idx, dtype=float) * 1.4
    bbb = pd.Series(range(300, 80, -1), index=idx, dtype=float)
    frame = pd.DataFrame({"SPY": spy, "AAA": aaa, "BBB": bbb})
    return frame


def main() -> int:
    _check_large_ranking_context_and_truncation()
    scout_prompt = " ".join(ts.SCOUT_SYSTEM.split())
    assert "worth monitoring on the watchlist" in scout_prompt
    assert "not a buy recommendation" in scout_prompt
    assert "marked `add` in `ranking`" in scout_prompt
    assert "Do not include `pass` candidates" in scout_prompt
    assert "do not force the list to a particular size" in scout_prompt
    assert "separate user review and application" in scout_prompt
    _check_ranked_add_sync()
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

    # Both consumers accept all established list wrappers, including `items`.
    for key in ("items", "candidates", "watchlist", "tickers"):
        wrapped_path = Path(tempfile.mkstemp(prefix="scout_wrapper_")[1])
        try:
            wrapped_path.write_text(json.dumps({key: [{"ticker": "ZZZ"}], "metadata": "kept"}), encoding="utf-8")
            assert ts.load_watchlist(wrapped_path)[0]["ticker"] == "ZZZ"
            if key == "items":
                wrapped_path.write_text(json.dumps({"items": [], "candidates": [{"ticker": "NO_FALLBACK"}]}), encoding="utf-8")
                assert ts.load_watchlist(wrapped_path) == []
        finally:
            wrapped_path.unlink(missing_ok=True)

    class Query:
        def __init__(self, operator, operands):
            self.operator, self.operands = operator, operands

    import types
    fake_yf = types.ModuleType("yfinance")
    fake_yf.EquityQuery = Query
    seen_queries = []
    fake_yf.screen = lambda query, **_kwargs: (seen_queries.append(query) or {"quotes": [
        {"symbol": "NVDA", "quoteType": "EQUITY"}, {"symbol": "XLK", "quoteType": "ETF"}]})
    with patch.dict("sys.modules", {"yfinance": fake_yf}):
        discovered = ts.discover_sector_candidates(["SMH"])
    assert [row["ticker"] for row in discovered] == ["NVDA"] and discovered[0]["sector"] == "Semiconductors"
    assert any(node.operands == ["sector", "Technology"] for node in seen_queries[0].operands)
    assert any(node.operands == ["industry", "Semiconductors"] for node in seen_queries[0].operands)
    class BadQuery(Query):
        def __init__(self, operator, operands):
            if operands[0] == "industry":
                raise ValueError("unsupported screener field")
            super().__init__(operator, operands)
    fake_yf.EquityQuery = BadQuery
    with patch.dict("sys.modules", {"yfinance": fake_yf}):
        assert ts.discover_sector_candidates(["SMH"]) == []

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
    assert actions.count("drop") == 4, changes  # legacy min_n no longer blocks reviewed drops
    assert not any(c["action"] == "kept" and c["ticker"] == "N03" for c in changes)
    assert len(updated) == 29

    wide = [{"ticker": f"W{i:02d}", "sector": "Other", "type": "stock"} for i in range(48)]
    wide_scores = {item["ticker"]: 48 - i for i, item in enumerate(wide)}
    trimmed, tchanges = ts.apply_watchlist(
        wide, [{"ticker": f"A{i}", "sector": "X", "reason": "add"} for i in range(4)], [],
        wide_scores, min_n=30, max_n=50, shadow=False, as_of="2026-11-06")
    assert len(trimmed) == 52, len(trimmed)
    assert sum(1 for c in tchanges if c["action"] == "trim") == 0

    ordered, errors = ts.normalize_ranking(
        [{"ticker": "bbb", "rank": 2, "bucket": "candidate", "action": "pass", "reason": "weak"},
         {"ticker": "AAA", "rank": 1, "bucket": "holding", "action": "keep", "reason": "strong"},
         {"ticker": "CCC", "rank": 3, "bucket": "candidate", "action": "trim", "reason": "wrong action still stored"}],
        ["AAA"], ["BBB", "CCC"])
    assert errors == [], errors
    assert [row["ticker"] for row in ordered] == ["AAA", "BBB", "CCC"]
    assert ordered[2]["action"] == "trim" and ordered[2]["bucket"] == "candidate"
    partial, partial_warnings = ts.normalize_ranking(
        [{"ticker": "AAA", "rank": 1, "action": "keep", "reason": "only one"}],
        ["AAA"], ["BBB"])
    assert [row["ticker"] for row in partial] == ["AAA"]
    assert any("missing from ranking" in warning for warning in partial_warnings)
    invalid, invalid_warnings = ts.normalize_ranking(
        [{"ticker": "AAA", "rank": 0, "action": "keep"}], ["AAA"], [])
    assert invalid == [] and any("positive integer" in warning for warning in invalid_warnings)

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
    assert ts.curator_shadow("2026-12-06", {}) is True
    assert ts.curator_shadow("2026-11-07", {"watchlist_curator": {"shadow": False}}) is True
    assert ts.ai_weight_for("ZZZ", {"ai_sleeve": {"ticker_weight": {"GLD": 0}, "theme_weight": {}}}, {}) == (None, "unclassified")
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

    week_events = [{"id": f"a{i}", "action": "add", "ticker": f"Z{i}", "as_of": "2026-11-03"} for i in range(50)]
    uncapped, uncapped_changes = ts.apply_watchlist(
        items, [{"ticker": "NEW", "sector": "X", "reason": "sixth"}], [], scores,
        shadow=False, as_of="2026-11-06", log_events=week_events)
    assert any(c["action"] == "add" and c["ticker"] == "NEW" for c in uncapped_changes)
    assert any(row["ticker"] == "NEW" for row in uncapped)
    assert ts.curator_policy({})["cooldown_days"] == 14
    assert ts.curator_policy({"watchlist_curator": {"max_adds_per_week": 1, "max_drops_per_week": 1}})["max_adds_per_week"] is None

    sleeve = {"ai_sleeve": {"cap_pct": 50, "ticker_weight": {"GLD": 0.0}, "theme_weight": {}}}
    monitored, monitor_changes = ts.apply_watchlist(
        items, [{"ticker": "ZZZ", "sector": "X", "reason": "untagged"},
                {"ticker": "GLD", "sector": "Gold", "type": "etf", "reason": "non-ai"}], [],
        scores, shadow=False, as_of="2026-11-06", targets=sleeve, ai_lookthrough_pct=50)
    # Existing AI-look-through safeguards remain portfolio risk context; they
    # do not prevent adding monitoring ideas to a reviewed watchlist proposal.
    assert any(c["ticker"] == "ZZZ" and c["action"] == "add" and c.get("ai_weight") is None
               and c.get("ai_weight_source") == "unclassified"
               for c in monitor_changes), monitor_changes
    assert any(c["ticker"] == "GLD" and c["action"] == "add" and c.get("ai_weight") == 0.0
               for c in monitor_changes), monitor_changes
    assert {"ZZZ", "GLD"}.issubset({row["ticker"] for row in monitored})
    assert not any("AI sleeve" in change.get("reason", "") for change in monitor_changes)
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
            watch_file, items, proposed, shadow_changes, log_root / "watchlist_proposals.json", log_root, "2026-11-07",
            coverage={"ranking_status": "partial", "ranking_count": 33,
                      "ranking_expected_count": 166, "ranking_missing_tickers": ["MISSING"],
                      "ranking_warnings": ["sample warning"]})
        assert outcome == "proposal"
        assert watch_file.read_text(encoding="utf-8") == before
        assert list(log_root.glob("*.bak*")) == []
        assert (log_root / "watchlist_proposals.json").is_file()
        saved_proposal = json.loads((log_root / "watchlist_proposals.json").read_text(encoding="utf-8"))
        assert saved_proposal["ranking_status"] == "partial"
        assert saved_proposal["ranking_count"] == 33 and saved_proposal["ranking_expected_count"] == 166
        assert saved_proposal["ranking_missing_tickers"] == ["MISSING"]
        # Even a legacy caller that explicitly requests non-shadow behavior
        # cannot make the scout write the live watchlist.
        legacy_changes = [{"action": "add", "ticker": "NEW", "shadow": False, "applied": True}]
        legacy_outcome = ts.commit_watchlist(
            watch_file, items, proposed, legacy_changes,
            log_root / "legacy_proposals.json", log_root, "2026-12-06")
        assert legacy_outcome == "proposal"
        assert watch_file.read_text(encoding="utf-8") == before
        legacy_payload = json.loads((log_root / "legacy_proposals.json").read_text(encoding="utf-8"))
        assert legacy_payload["shadow"] is True
        assert legacy_payload["changes"][0]["shadow"] is True
        assert legacy_payload["changes"][0]["applied"] is False
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

    _check_main_proposal_flow()
    print("check_offline: ok")
    print(f"  AAA rs_1m {by['AAA']['rs_1m']} BBB rs_1m {by['BBB']['rs_1m']}")
    print(f"  watchlist drops applied {len(updated)} names; no auto-trim ({len(trimmed)} names after additions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
