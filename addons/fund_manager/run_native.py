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
_calls: list = []


def _addons_dir() -> Path:
    addons = Path("/home/appuser/addons")
    if not (addons / "cost_ledger.py").is_file():
        addons = Path(__file__).resolve().parents[1]
    if str(addons) not in sys.path:
        sys.path.insert(0, str(addons))
    return addons


def _usage_bits(response) -> tuple[str | None, int, int, float | None]:
    """Model, tokens, and provider cost from one LangChain completion, when present."""
    model, tokens_in, tokens_out, cost = None, 0, 0, None
    try:
        gen = response.generations[0][0]
        msg = getattr(gen, "message", None)
        meta = getattr(msg, "response_metadata", None) or {}
        usage = getattr(msg, "usage_metadata", None) or {}
        if isinstance(meta, dict):
            model = meta.get("model_name") or meta.get("model")
            blob = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
            token_blob = meta.get("token_usage") if isinstance(meta.get("token_usage"), dict) else {}
            cost = blob.get("cost", meta.get("cost"))
            tokens_in = int(blob.get("prompt_tokens") or token_blob.get("prompt_tokens") or 0)
            tokens_out = int(blob.get("completion_tokens") or token_blob.get("completion_tokens") or 0)
        if isinstance(usage, dict):
            tokens_in = int(usage.get("input_tokens") or usage.get("prompt_tokens") or tokens_in or 0)
            tokens_out = int(usage.get("output_tokens") or usage.get("completion_tokens") or tokens_out or 0)
            if cost is None:
                cost = usage.get("cost")
            model = model or usage.get("model")
    except Exception:
        pass
    return model, tokens_in, tokens_out, cost


def _install_cost_hook() -> None:
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
        global _completions
        orig_end(self, response, **kwargs)
        _completions += 1
        model, tokens_in, tokens_out, provider_cost = _usage_bits(response)
        try:
            _addons_dir()
            import cost_ledger
            _calls.append(cost_ledger.price_call(model, tokens_in, tokens_out, provider_cost,
                                                 fallback_model=cost_ledger.DEFAULT_FALLBACK_MODEL))
        except Exception:
            _calls.append({
                "model": model, "tokens_in": tokens_in, "tokens_out": tokens_out,
                "cost_usd": None, "cost_source": "unpriced",
            })

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
    priced = [call for call in _calls if isinstance(call.get("cost_usd"), (int, float))]
    sources = {call.get("cost_source") for call in _calls}
    if sources == {"provider"}:
        source = "provider"
    elif "provider" in sources and sources & {"price_table", "price_table_fallback"}:
        source = "mixed"
    elif sources & {"price_table", "price_table_fallback"}:
        source = "price_table" if "price_table_fallback" not in sources else "price_table_fallback"
    else:
        source = "unpriced"
    tokens_in = sum(int(call.get("tokens_in") or 0) for call in _calls) or stats.get("tokens_in")
    tokens_out = sum(int(call.get("tokens_out") or 0) for call in _calls) or stats.get("tokens_out")
    cost = round(sum(float(call["cost_usd"]) for call in priced), 6) if priced else None
    rec = {
        "ticker": ticker,
        "analysis_date": analysis,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completions": _completions,
        "llm_calls_handler": stats.get("llm_calls"),
        "tool_calls": stats.get("tool_calls"),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost,
        "cost_source": source,
        "calls": _calls,
        "cost_note": (
            "USD cost uses provider usage.cost when the payload has it. Otherwise each completion "
            "is priced from addons/openrouter_models.json. Upstream is not edited, so native runs "
            "do not send OpenRouter usage.include. llm_calls_handler can double-count if both "
            "on_llm_start and on_chat_model_start fire; completions counts on_llm_end."
        ),
    }
    print(
        f"RUN_COST ticker={ticker} date={analysis} completions={_completions} "
        f"tokens_in={rec['tokens_in']} tokens_out={rec['tokens_out']} cost_usd={rec['cost_usd']}",
        flush=True,
    )
    try:
        _addons_dir()
        import cost_ledger
        root = Path(os.environ.get("FM_DATA_ROOT", "/home/appuser/.tradingagents"))
        cost_ledger.write_run_cost(root, rec)
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


def _install_macro_context() -> None:
    """Append the saved theme-scout note to every agent's instrument context."""
    raw = os.environ.get("ARGUS_MACRO_CONTEXT_FILE", "").strip()
    if not raw:
        return
    path = Path(raw)
    if not path.is_file():
        print(f"[run_native] macro context file missing: {path}", flush=True)
        raise SystemExit(2)
    text = path.read_text(encoding="utf-8").strip()
    if len(text) > 12000:
        text = text[:12000] + "\n…[theme scout context truncated]"
    block = "\n\n## Theme scout (macro ranking for this book)\n" + text
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    orig = TradingAgentsGraph.create_run_state

    def wrapped(self, company_name, trade_date, asset_type="stock", portfolio=None):
        state = orig(self, company_name, trade_date, asset_type, portfolio)
        state["instrument_context"] = (state.get("instrument_context") or "") + block
        return state

    TradingAgentsGraph.create_run_state = wrapped  # type: ignore[method-assign]
    print(f"[run_native] macro context attached ({len(text)} chars)", flush=True)


_install_macro_context()


if __name__ == "__main__":
    app()