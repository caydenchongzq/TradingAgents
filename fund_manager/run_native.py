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


if __name__ == "__main__":
    app()