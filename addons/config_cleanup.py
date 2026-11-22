"""Preview and apply the narrowly scoped Argus configuration cleanup.

Only changed key paths are printed. Portfolio books, credentials, historical
reports and append-only watchlist events are never rewritten by this command.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class CleanupError(ValueError):
    pass


def _digest(data: bytes | None) -> str:
    return "missing" if data is None else hashlib.sha256(data).hexdigest()


def clean_targets(value: dict) -> tuple[dict, list[str]]:
    """Preserve effective policy/unknown metadata while dropping proven no-ops."""
    from fund_manager.trade_validation import parse_trade_policy

    if not isinstance(value, dict):
        raise CleanupError("Targets must be an object")
    result = deepcopy(value)
    changed = []
    for key in ("weights_pct", "is_guidance"):
        if key in result:
            del result[key]
            changed.append(key)
    curator = result.get("watchlist_curator")
    if curator is not None and not isinstance(curator, dict):
        raise CleanupError("watchlist_curator must be an object")
    if isinstance(curator, dict):
        for key in ("cooldown_days", "max_adds_per_week", "max_drops_per_week", "shadow", "shadow_until"):
            if key in curator:
                del curator[key]
                changed.append(f"watchlist_curator.{key}")
    effective = parse_trade_policy(result)
    policy = result.get("trade_policy", {})
    if not isinstance(policy, dict):
        raise CleanupError("trade_policy must be an object")
    policy = dict(policy)
    for key, default in effective.items():
        if key not in policy:
            policy[key] = default
            changed.append(f"trade_policy.{key}")
    result["trade_policy"] = policy
    return result, changed


def clean_watchlist(value) -> tuple[object, list[str]]:
    """Keep wrapper, order and unknown metadata; strip only current row tags."""
    result = deepcopy(value)
    prefix = ""
    if isinstance(result, list):
        rows = result
    elif isinstance(result, dict):
        key = next((key for key in ("items", "candidates", "watchlist", "tickers")
                    if isinstance(result.get(key), list)), None)
        if key is None:
            raise CleanupError("Watchlist object has no supported list")
        prefix = key + "."
        rows = result[key]
    else:
        raise CleanupError("Watchlist must be a list or supported wrapper")
    changed = []
    for index, row in enumerate(rows):
        if isinstance(row, dict) and "sub_theme" in row:
            del row["sub_theme"]
            changed.append(f"{prefix}[{index}].sub_theme")
    return result, changed


def _read_json(path: Path) -> tuple[bytes | None, object]:
    if not path.exists():
        return None, None
    try:
        data = path.read_bytes()
        def reject_constant(_):
            raise ValueError("Nonfinite JSON")
        parsed = json.loads(data, parse_constant=reject_constant)
        return data, parsed
    except (OSError, ValueError) as exc:
        # Do not echo parsing errors containing user values.
        raise CleanupError(f"Cannot read valid JSON: {path.name}") from exc


def preview_cleanup(data_root: Path) -> dict:
    data_root = Path(data_root).resolve()
    entries = []
    versions = {}
    for name, transform in (("targets.json", clean_targets), ("watchlist.json", clean_watchlist)):
        path = data_root / name
        original, parsed = _read_json(path)
        versions[name] = _digest(original)
        if original is None:
            continue
        try:
            cleaned, keys = transform(parsed)
            after = (json.dumps(cleaned, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
        except (TypeError, ValueError) as exc:
            raise CleanupError(f"Cannot clean {name}: invalid configuration") from exc
        if keys:
            entries.append({"path": path, "before": original, "after": after, "changed_keys": keys})
    fingerprint = hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()
    return {"data_root": data_root, "fingerprint": fingerprint, "versions": versions, "entries": entries}


@contextmanager
def _write_lock(data_root: Path):
    data_root.mkdir(parents=True, exist_ok=True)
    with (data_root / ".argus-run.lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                lock.seek(0)
                if not lock.read(1):
                    lock.write(b"0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise CleanupError("Portfolio data is busy; try after the analysis/write finishes") from exc
        try:
            # A pending watchlist transaction owns the current list. Let Argus
            # recover it before cleanup rather than rewriting journal contents.
            if (data_root / ".watchlist_transaction.json").exists():
                raise CleanupError("Pending watchlist transaction must be recovered before cleanup")
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _atomic_write(path: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        if path.exists():
            os.chmod(temporary, path.stat().st_mode & 0o777)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def apply_cleanup(data_root: Path, expected_fingerprint: str) -> dict:
    if not expected_fingerprint:
        raise CleanupError("Apply needs the fingerprint from a reviewed preview")
    root = Path(data_root).resolve()
    with _write_lock(root):
        preview = preview_cleanup(root)
        if preview["fingerprint"] != expected_fingerprint:
            raise CleanupError("Configuration changed after preview; review a fresh preview")
        entries = preview["entries"]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backups = []
        for entry in entries:
            backup = entry["path"].with_name(entry["path"].name + ".bak_" + stamp)
            mode = entry["path"].stat().st_mode & 0o777
            descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                os.chmod(backup, mode)
                stream.write(entry["before"])
                stream.flush()
                os.fsync(stream.fileno())
            backups.append(backup)
        written = []
        try:
            # All JSON and backups have been validated before the first write.
            # Recheck every input once more against non-cooperating editors.
            for name, version in preview["versions"].items():
                path = root / name
                if _digest(path.read_bytes() if path.exists() else None) != version:
                    raise CleanupError("Configuration changed while preparing cleanup")
            for entry in entries:
                _atomic_write(entry["path"], entry["after"])
                written.append(entry)
        except Exception as exc:
            failures = []
            for entry in reversed(written):
                try:
                    _atomic_write(entry["path"], entry["before"])
                except Exception:
                    failures.append(entry["path"].name)
            if failures:
                raise CleanupError("Cleanup failed; manual backup restore needed for " + ", ".join(failures)) from exc
            raise CleanupError("Cleanup failed; original files restored (backups retained)") from exc
        return {"fingerprint": preview["fingerprint"], "changed": [
            {"file": entry["path"].name, "keys": entry["changed_keys"], "backup": backup.name}
            for entry, backup in zip(entries, backups)]}


def public_preview(preview: dict) -> dict:
    return {"fingerprint": preview["fingerprint"], "changes": [
        {"file": entry["path"].name, "keys": entry["changed_keys"]} for entry in preview["entries"]]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expected-fingerprint", default="")
    args = parser.parse_args(argv)
    try:
        result = (apply_cleanup(args.data_root, args.expected_fingerprint) if args.apply
                  else public_preview(preview_cleanup(args.data_root)))
    except (CleanupError, OSError) as exc:
        print(str(exc))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
