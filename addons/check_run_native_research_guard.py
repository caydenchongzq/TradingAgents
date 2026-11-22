"""Check the native wrapper's auxiliary-context and prompt-mode boundary offline."""
from __future__ import annotations

import os
import runpy
import sys
import types
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parent / "fund_manager" / "run_native.py"


def run_wrapper(argv: list[str], inherited_research: str | None) -> tuple[int | None, list[str], int]:
    installed_modes: list[str] = []
    app_calls = 0

    def app():
        nonlocal app_calls
        app_calls += 1

    class Stats:
        def __init__(self, *args, **kwargs):
            pass

        def on_llm_end(self, *args, **kwargs):
            pass

        def get_stats(self):
            return {}

    tradingagents = types.ModuleType("tradingagents")
    tradingagents.__path__ = []
    defaults = types.ModuleType("tradingagents.default_config")
    defaults.DEFAULT_CONFIG = {}
    cli = types.ModuleType("cli")
    cli.__path__ = []
    cli_main = types.ModuleType("cli.main")
    cli_main.app = app
    cli_stats = types.ModuleType("cli.stats_handler")
    cli_stats.StatsCallbackHandler = Stats
    agent_config = types.ModuleType("agent_config")
    agent_config.install_native_seats = lambda _config: installed_modes.append(
        os.environ.get("ARGUS_RESEARCH_ONLY", "")
    )
    modules = {
        "tradingagents": tradingagents,
        "tradingagents.default_config": defaults,
        "cli": cli,
        "cli.main": cli_main,
        "cli.stats_handler": cli_stats,
        "agent_config": agent_config,
    }
    env = {"ARGUS_MACRO_CONTEXT_FILE": "/does/not/exist/macro-context.md"}
    if inherited_research is not None:
        env["ARGUS_RESEARCH_ONLY"] = inherited_research
    output = None
    with patch.dict(sys.modules, modules), patch.dict(os.environ, env, clear=False):
        # Each case starts from the requested inherited state.
        if inherited_research is None:
            os.environ.pop("ARGUS_RESEARCH_ONLY", None)
        with patch.object(sys, "argv", [str(SCRIPT), *argv]):
            try:
                runpy.run_path(str(SCRIPT), run_name="__main__")
            except SystemExit as exc:
                output = exc.code
    return output, installed_modes, app_calls


def main() -> int:
    # A direct CLI flag sets the guard before agent_config installs seat hooks,
    # then rejects the saved macro context before opening the path or starting app().
    code, installed, app_calls = run_wrapper(["--research-only"], None)
    assert code == 2, code
    assert installed == ["1"], installed
    assert app_calls == 0, app_calls

    # An inherited managed-launch flag enforces the same auxiliary-context rule.
    code, installed, app_calls = run_wrapper([], "1")
    assert code == 2, code
    assert installed == ["1"], installed
    assert app_calls == 0, app_calls

    print("run_native research guard offline check: ok (direct flag, inherited flag, macro context rejected before app)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
