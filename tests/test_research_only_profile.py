"""Account-blind native profile contract and report handoff."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from typer.testing import CliRunner

import tradingagents.agents.analysts.market_analyst as market
import cli.main as cli_main
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.rating import run_rating
from tradingagents.agents.schemas import (
    ConditionalLevel,
    InstrumentOutlook,
    InstrumentResearchBrief,
    InstrumentSentiment,
    InstrumentValuation,
    ValuationRange,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.reporting import write_report_tree


def _brief() -> InstrumentResearchBrief:
    return InstrumentResearchBrief(
        outlook=InstrumentOutlook.ATTRACTIVE,
        thesis="Revenue growth and margins support the thesis.",
        counter_thesis="Valuation leaves little room for execution misses.",
        valuation=InstrumentValuation(
            target_price=210,
            target_range=ValuationRange(low=190, high=225),
            method="DCF",
            assumptions="Revenue growth moderates over three years.",
        ),
        thesis_horizon="12 to 18 months, through the next three reported quarters",
        catalysts=["Quarterly results"],
        risks=["Multiple compression"],
        sentiment=InstrumentSentiment(summary="Mixed social and news tone", as_of="2026-11-17"),
        conditional_levels=[ConditionalLevel(condition="holds support", level=180, meaning="technical confirmation")],
        confidence="medium",
        unknowns=["Future demand durability"],
        evidence_dates=[{"source": "Quarterly filing", "as_of": "2026-10-28"}],
    )


@pytest.mark.unit
def test_research_only_portfolio_manager_emits_structured_brief_without_portfolio_values():
    structured = MagicMock()
    structured.invoke.return_value = _brief()
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    node = create_portfolio_manager(llm, research_only=True)
    state = {
        "company_of_interest": "NVDA",
        "instrument_context": "NVDA is a semiconductor company.",
        "portfolio_context": "SENTINEL HOLDINGS 37 shares; SENTINEL CASH $5,000",
        "investment_plan": "Research plan",
        "trader_investment_plan": "Instrument hypothesis",
        "risk_debate_state": {
            "history": "risk debate",
            "aggressive_history": "upside",
            "conservative_history": "downside",
            "neutral_history": "balanced",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "count": 1,
        },
    }

    result = node(state)

    prompt = structured.invoke.call_args.args[0]
    assert "SENTINEL" not in prompt
    assert result["final_rating"] == "RESEARCH"
    assert result["research_brief"]["valuation"]["target_price"] == 210
    assert result["research_brief"]["thesis_horizon"].startswith("12 to 18 months")
    assert "position_sizing" not in result["research_brief"]
    assert run_rating({"research_only": True, "final_rating": "REVIEW"}) == "Research"


@pytest.mark.unit
def test_research_only_report_writes_versioned_brief_and_cleans_stale_brief(tmp_path):
    state = {
        "research_only": True,
        "trade_date": "2026-11-17",
        "final_trade_decision": "Instrument research summary",
        "research_brief": _brief().model_dump(mode="json"),
    }
    write_report_tree(state, "NVDA", tmp_path, html=False)

    artifact = json.loads((tmp_path / "research_brief.json").read_text(encoding="utf-8"))
    assert artifact["schema_version"] == 1
    assert artifact["profile"] == "research_only"
    assert artifact["ticker"] == "NVDA"
    assert artifact["analysis_date"] == "2026-11-17"
    assert artifact["generated_utc"].endswith("Z")
    assert artifact["valuation"]["target_range"] == {"low": 190.0, "high": 225.0}
    assert artifact["thesis_horizon"].startswith("12 to 18 months")

    write_report_tree({"research_only": True, "trade_date": "2026-11-18"}, "NVDA", tmp_path, html=False)
    assert not (tmp_path / "research_brief.json").exists()


@pytest.mark.unit
def test_standard_report_does_not_write_research_brief(tmp_path):
    write_report_tree({"final_trade_decision": "BUY"}, "NVDA", tmp_path, html=False)
    assert not (tmp_path / "research_brief.json").exists()


@pytest.mark.unit
def test_graph_rejects_portfolio_input_in_research_only_mode():
    graph = object.__new__(TradingAgentsGraph)
    graph.research_only = True

    with pytest.raises(ValueError, match="cannot receive portfolio context"):
        graph.create_run_state("NVDA", "2026-11-17", portfolio=SimpleNamespace())


@pytest.mark.unit
def test_research_only_does_not_read_or_write_portfolio_memory():
    graph = object.__new__(TradingAgentsGraph)
    graph.research_only = True
    graph.memory_log = MagicMock()
    graph.settle_all_pending = MagicMock()
    graph._log_state = MagicMock()

    assert graph._memory_step({"company_of_interest": "NVDA", "trade_date": "2026-11-17"}) == {
        "past_context": "", "memory_note": "",
    }
    graph.record_decision("NVDA", "2026-11-17", {"final_trade_decision": "research"})

    graph.memory_log.get_past_context.assert_not_called()
    graph.memory_log.store_decision.assert_not_called()
    graph.settle_all_pending.assert_not_called()
    graph._log_state.assert_called_once()


@pytest.mark.unit
def test_research_only_market_prompt_describes_atr_without_sizing(monkeypatch):
    captured = {}

    def capture_prompt(prompt, _llm, _tools, messages):
        captured["prompt"] = "\n".join(message.content for message in prompt.format_messages(messages=messages))
        return AIMessage(content="market report"), "market report"

    monkeypatch.setattr(market, "take_turn", capture_prompt)
    node = market.create_market_analyst(MagicMock(), research_only=True)
    node({
        "company_of_interest": "NVDA",
        "trade_date": "2026-11-17",
        "instrument_context": "NVDA semiconductor company",
        "messages": [],
    })

    prompt = captured["prompt"]
    assert "Describe ATR as a measure of observed instrument volatility" in prompt
    assert "do not derive user-specific stops or position sizes" in prompt
    assert "Do not recommend user-specific orders, share quantities, stops, or allocations" in prompt
    assert "adjust position sizes based on current market volatility" not in prompt


@pytest.mark.unit
def test_cli_selects_research_only_and_rejects_portfolio(monkeypatch):
    calls = []
    monkeypatch.setattr(cli_main, "run_analysis", lambda **kwargs: calls.append(kwargs))
    runner = CliRunner()

    result = runner.invoke(cli_main.app, [
        "--research-only", "--ticker", "NVDA", "--date", "2026-11-17",
    ])
    assert result.exit_code == 0
    assert calls[0]["flags"]["research_only"] is True

    rejected = runner.invoke(cli_main.app, ["--research-only", "--portfolio", "sentinel.json"])
    assert rejected.exit_code == 1
    assert "cannot be combined with --portfolio" in rejected.output
    assert len(calls) == 1
