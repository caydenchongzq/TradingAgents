#!/usr/bin/env python3
"""Native TradingAgents run with Yahoo news primary and Alpha Vantage fallback.

Lives in the fund_manager add-on so the project's own files stay untouched.
Mutates DEFAULT_CONFIG before the CLI builds its run config, then hands off to
the normal `tradingagents` Typer entrypoint (same flags as `docker compose run
tradingagents --ticker ...`).
"""
from __future__ import annotations

from tradingagents.default_config import DEFAULT_CONFIG

# Yahoo first; Alpha Vantage only if Yahoo cannot serve the news call.
vendors = DEFAULT_CONFIG.setdefault("data_vendors", {})
vendors["news_data"] = "yfinance,alpha_vantage"

from cli.main import app  # noqa: E402  — after the override above

import atexit
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Token totals are already collected by the CLI's StatsCallbackHandler, but nothing
# writes them down. Cost in USD is logged only when the provider payload carries it
# (OpenRouter does not, unless the client asks for usage.include, which lives in
# upstream client code this add-on does not edit).
_handlers: list = []
_completions = 0
_cost_usd = 0.0
_saw_cost = False


def _install_cost_hook() -> None:
    global _completions, _cost_usd, _saw_cost
    try:
        from cli.stats_handler import StatsCallbackHandler
    except Exception as exc:  # noqa: BLE001
        print(f"[run_native] cost hook unavailable: {exc}", flush=True)
        return

    orig_init = StatsCallbackHandler.__init__
    orig_end = StatsCallbackHandler.on_llm_end

    def _init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        _handlers.append(self)

    def _end(self, response, **kwargs):
        global _completions, _cost_usd, _saw_cost
        orig_end(self, response, **kwargs)
        _completions += 1
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            meta = getattr(msg, "response_metadata", None) or {}
            usage = getattr(msg, "usage_metadata", None) or {}
            cost = None
            if isinstance(meta, dict):
                blob = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
                cost = blob.get("cost", meta.get("cost"))
            if cost is None and isinstance(usage, dict):
                cost = usage.get("cost")
            if cost is not None:
                _cost_usd += float(cost)
                _saw_cost = True
        except Exception:
            pass

    StatsCallbackHandler.__init__ = _init  # type: ignore[method-assign]
    StatsCallbackHandler.on_llm_end = _end  # type: ignore[method-assign]


def _flag(name: str) -> str | None:
    argv = sys.argv[1:]
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


def _write_cost() -> None:
    ticker, analysis = _flag("--ticker"), _flag("--date")
    if not ticker and not _handlers:
        return
    stats = _handlers[-1].get_stats() if _handlers else {}
    rec = {
        "ticker": ticker,
        "analysis_date": analysis,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completions": _completions,
        "llm_calls_handler": stats.get("llm_calls"),
        "tool_calls": stats.get("tool_calls"),
        "tokens_in": stats.get("tokens_in"),
        "tokens_out": stats.get("tokens_out"),
        "cost_usd": round(_cost_usd, 6) if _saw_cost else None,
        "cost_note": (
            "USD cost summed from provider response metadata."
            if _saw_cost else
            "USD cost was not in the provider payload. Token totals are from the in-process "
            "StatsCallbackHandler. Upstream does not send OpenRouter usage.include, and that "
            "client is not edited from this add-on. llm_calls_handler can double-count if both "
            "on_llm_start and on_chat_model_start fire; completions counts on_llm_end."
        ),
    }
    print(
        f"RUN_COST ticker={ticker} date={analysis} completions={_completions} "
        f"tokens_in={rec['tokens_in']} tokens_out={rec['tokens_out']} cost_usd={rec['cost_usd']}",
        flush=True,
    )
    try:
        out = Path("/home/appuser/.tradingagents/fund_manager/run_costs")
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%H%M%S")
        name = f"{ticker or 'unknown'}_{(analysis or 'nodate').replace(':','-')}_{stamp}.json"
        (out / name).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"[run_native] could not write cost file: {exc}", flush=True)


_install_cost_hook()
atexit.register(_write_cost)


def _boot_agent_config() -> None:
    """Load addons/agent_config.json and patch seat factories. No model call."""
    addons = Path("/home/appuser/addons")
    if not (addons / "agent_config.py").is_file():
        addons = Path(__file__).resolve().parents[1]
    if str(addons) not in sys.path:
        sys.path.insert(0, str(addons))
    try:
        import agent_config
        agent_config.install_native_seats(DEFAULT_CONFIG)
    except Exception as exc:  # noqa: BLE001
        print(f"[run_native] agent config was not installed: {exc}", flush=True)


_boot_agent_config()


if __name__ == "__main__":
    app()