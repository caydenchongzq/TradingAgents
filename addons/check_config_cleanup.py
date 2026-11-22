"""Offline proof of narrow cleanup, preview conflicts and backup recovery."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

import config_cleanup as cleanup


def expect_error(call, text):
    try:
        call()
    except cleanup.CleanupError as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError("Expected cleanup error")


def main() -> int:
    fixture = {
        "weights_pct": {"PRIVATE": 23}, "is_guidance": True, "_note": "retain",
        "budget": {"enabled": False, "weekly_usd": 29, "native_run_estimate_usd": 2},
        "weekly_budget_usd": 35,
        "portfolio_guidance": {"cash_reference_pct": 10, "long_term_pct": 30, "swing_pct": 60},
        "ai_sleeve": {"source": "keep provenance", "ticker_weight": {"SPY": .33}},
        "watchlist_curator": {"max_candidates": 100, "max_stocks_per_sector": 10,
                              "shadow": True, "shadow_until": "old", "cooldown_days": 28,
                              "max_adds_per_week": 5, "max_drops_per_week": None},
        "unknown": {"custom": "preserve"},
    }
    original = deepcopy(fixture)
    cleaned, keys = cleanup.clean_targets(fixture)
    assert fixture == original
    assert "weights_pct" not in cleaned and "is_guidance" not in cleaned
    for key in ("budget", "weekly_budget_usd", "portfolio_guidance", "ai_sleeve", "unknown", "_note"):
        assert cleaned[key] == original[key], key
    assert cleaned["watchlist_curator"] == {"max_candidates": 100, "max_stocks_per_sector": 10}
    assert cleaned["trade_policy"] == {"fee_per_order_usd": 3, "min_buy_notional_usd": 1000,
                                      "min_residual_notional_usd": 1000}
    assert cleanup.clean_targets(cleaned)[1] == []
    for wrapper in (None, "items", "candidates", "watchlist", "tickers"):
        rows = ["PLAIN", {"ticker": "PRIVATE", "sub_theme": "ai_compute_semis", "note": "retain"}]
        before = rows if wrapper is None else {wrapper: rows, "meta": "retain"}
        after, changes = cleanup.clean_watchlist(before)
        after_rows = after if wrapper is None else after[wrapper]
        assert after_rows == ["PLAIN", {"ticker": "PRIVATE", "note": "retain"}]
        assert len(changes) == 1
        assert "sub_theme" in rows[1]
        assert cleanup.clean_watchlist(after)[1] == []
    with tempfile.TemporaryDirectory(prefix="argus-cleanup-proof-") as folder:
        root = Path(folder)
        targets, watch = root / "targets.json", root / "watchlist.json"
        targets.write_text(json.dumps(fixture))
        watch.write_text(json.dumps({"items": [{"ticker": "PRIVATE", "sub_theme": "old"}], "meta": 1}))
        targets.chmod(0o600)
        watch.chmod(0o600)
        original_targets, original_watch = targets.read_bytes(), watch.read_bytes()
        preview = cleanup.preview_cleanup(root)
        assert targets.read_bytes() == original_targets and watch.read_bytes() == original_watch
        assert not list(root.glob("*.bak_*"))
        public = json.dumps(cleanup.public_preview(preview))
        assert "PRIVATE" not in public and "preserve" not in public
        # Hashes cover changed and unchanged inputs; external edits stale the preview.
        watch.write_text(watch.read_text() + "\n")
        expect_error(lambda: cleanup.apply_cleanup(root, preview["fingerprint"]), "changed after preview")
        watch.write_bytes(original_watch)
        # All documents are parsed before any backup/write.
        watch.write_text("{")
        expect_error(lambda: cleanup.preview_cleanup(root), "Cannot read valid JSON")
        assert targets.read_bytes() == original_targets
        watch.write_bytes(original_watch)
        pending = root / ".watchlist_transaction.json"
        pending.write_text("{}")
        expect_error(lambda: cleanup.apply_cleanup(root, preview["fingerprint"]), "Pending watchlist transaction")
        pending.unlink()
        if os.name != "nt":
            import fcntl
            with (root / ".argus-run.lock").open("a+b") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                expect_error(lambda: cleanup.apply_cleanup(root, preview["fingerprint"]), "busy")
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        # Failure of the second file must roll the first one back exactly.
        real_write = cleanup._atomic_write
        count = 0
        def fail_second(path, data):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("fixture write failure")
            real_write(path, data)
        cleanup._atomic_write = fail_second
        try:
            expect_error(lambda: cleanup.apply_cleanup(root, preview["fingerprint"]), "original files restored")
        finally:
            cleanup._atomic_write = real_write
        assert targets.read_bytes() == original_targets and watch.read_bytes() == original_watch
        result = cleanup.apply_cleanup(root, preview["fingerprint"])
        assert len(result["changed"]) == 2
        assert (targets.stat().st_mode & 0o777) == 0o600
        assert (root / result["changed"][0]["backup"]).read_bytes() == original_targets
        assert (root / result["changed"][1]["backup"]).read_bytes() == original_watch
        for entry in result["changed"]:
            assert ((root / entry["backup"]).stat().st_mode & 0o777) == 0o600
        again = cleanup.preview_cleanup(root)
        assert again["entries"] == []
        assert cleanup.apply_cleanup(root, again["fingerprint"])["changed"] == []
        assert not (root / "my_book.json").exists()
        assert not (root / "watchlist_log.jsonl").exists()
    print("config cleanup offline check: passed (preservation, idempotence, lock/hash guards, rollback, backups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
