"""Append-only watchlist change log and undo.

The log is ``<data-root>/watchlist_log.jsonl``. Lines are never rewritten.
Undo appends a new line that points at the event it reverses.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


def log_path(data_root: Path | str) -> Path:
    return Path(data_root) / "watchlist_log.jsonl"


def read_events(data_root: Path | str) -> list[dict]:
    path = log_path(data_root)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def append_event(data_root: Path | str, event: dict) -> dict:
    path = log_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(event)
    row.setdefault("id", uuid.uuid4().hex)
    row.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def _day(value) -> datetime | None:
    if not value:
        return None
    text = str(value)[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return None


def undone_ids(events: list[dict]) -> set[str]:
    return {str(row.get("undo_of")) for row in events if row.get("action") == "undo" and row.get("undo_of")}


def active_events(events: list[dict]) -> list[dict]:
    undone = undone_ids(events)
    return [row for row in events if row.get("action") in {"add", "drop"} and str(row.get("id")) not in undone]


def cooldown_tickers(events: list[dict], as_of: str, days: int = 28) -> set[str]:
    """Tickers added or dropped inside the cooldown, ignoring ones already undone."""
    anchor = _day(as_of)
    if anchor is None:
        return set()
    start = anchor - timedelta(days=days)
    names = set()
    for row in active_events(events):
        when = _day(row.get("as_of") or row.get("ts"))
        ticker = str(row.get("ticker") or "").upper()
        if ticker and when is not None and start <= when <= anchor:
            names.add(ticker)
    return names


def week_counts(events: list[dict], as_of: str) -> dict[str, int]:
    anchor = _day(as_of)
    counts = {"add": 0, "drop": 0}
    if anchor is None:
        return counts
    monday = anchor - timedelta(days=anchor.weekday())
    sunday = monday + timedelta(days=6)
    for row in active_events(events):
        when = _day(row.get("as_of") or row.get("ts"))
        action = row.get("action")
        if when is not None and monday <= when <= sunday and action in counts:
            counts[action] += 1
    return counts


def last_undoable(events: list[dict]) -> dict | None:
    undone = undone_ids(events)
    for row in reversed(events):
        if row.get("action") in {"add", "drop"} and str(row.get("id")) not in undone:
            return row
    return None


def undo_items(items: list[dict], event: dict) -> list[dict]:
    """Return the list after reversing one add or drop. Does not touch the log."""
    ticker = str(event.get("ticker") or "").upper()
    rows = []
    for item in items or []:
        row = dict(item) if isinstance(item, dict) else {"ticker": item}
        row["ticker"] = str(row.get("ticker") or "").upper()
        if row["ticker"]:
            rows.append(row)
    if event.get("action") == "add":
        return [row for row in rows if row["ticker"] != ticker]
    if event.get("action") == "drop":
        if any(row["ticker"] == ticker for row in rows):
            return rows
        stored = event.get("row") if isinstance(event.get("row"), dict) else {"ticker": ticker}
        restored = dict(stored)
        restored["ticker"] = ticker
        rows.append(restored)
        return rows
    return rows
