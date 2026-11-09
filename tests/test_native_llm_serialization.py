"""Regression checks for warming native chat models before parallel graph calls."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor

from addons import agent_config
from langchain_core.language_models import BaseChatModel

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph import setup as graph_setup
from tradingagents.graph import trading_graph
from tradingagents.llm_clients.openai_client import NormalizedChatOpenAI
from tradingagents.memory.reflection import Reflector


class _Client:
    def __init__(self, model):
        self.model = model

    def get_llm(self):
        return self.model


def test_graph_construction_primes_tier_and_native_override_models(monkeypatch, tmp_path):
    """Tier models and run_native seat overrides are warm before graph workers start."""
    monkeypatch.setenv("ARGUS_AGENT_CONFIG", str(tmp_path / "empty-agent-config.json"))
    tier_models = {}
    override_models = []
    market_factory_entered = []

    def create_tier(config, tier, **_):
        model = NormalizedChatOpenAI(
            model=config[f"{tier}_think_llm"], api_key="offline-placeholder"
        )
        tier_models[tier] = model
        return _Client(model)

    def create_override(provider, model, **_):
        actual = NormalizedChatOpenAI(model=model, api_key="offline-placeholder")
        override_models.append(actual)
        return _Client(actual)

    monkeypatch.setattr(trading_graph, "create_tier_client", create_tier)
    monkeypatch.setattr("tradingagents.llm_clients.create_llm_client", create_override)

    # install_native_seats patches graph_setup factories process-wide. Restore
    # their original functions even if graph construction or an assertion fails.
    seat_factory_names = {name for _, name in agent_config._NATIVE_FACTORIES.values()}
    original_factories = {name: getattr(graph_setup, name) for name in seat_factory_names}
    original_reflector_init = Reflector.__init__
    original_market_factory = original_factories["create_market_analyst"]

    def observe_market_factory(llm):
        # This factory runs during TradingAgentsGraph construction, before any
        # analyst can be scheduled. Its shared quick client must already have
        # a serialized cache entry.
        assert isinstance(llm, agent_config._SeatCall)
        assert llm.inner is tier_models["quick"]
        assert "_serialized" in tier_models["quick"].__dict__
        market_factory_entered.append(True)
        return original_market_factory(llm)

    graph_setup.create_market_analyst = observe_market_factory
    try:
        agent_config.install_native_seats(DEFAULT_CONFIG)

        config = copy.deepcopy(DEFAULT_CONFIG)
        config.update(
            results_dir=str(tmp_path / "results"),
            data_cache_dir=str(tmp_path / "cache"),
            memory_log_path=str(tmp_path / "memory.md"),
            llm_provider="openrouter",
            quick_think_llm="openai/gpt-6-luna",
            deep_think_llm="openai/gpt-6.1-sol",
        )
        graph = trading_graph.TradingAgentsGraph(config=config)

        assert market_factory_entered
        assert all(isinstance(model, BaseChatModel) for model in tier_models.values())
        assert all("_serialized" in model.__dict__ for model in tier_models.values())
        # The default conservative-risk seat uses its own OpenRouter client,
        # created by the run_native seat wrapper during graph construction.
        assert len(override_models) == 1
        assert override_models[0].model_name == "anthropic/claude-opus-5.5"
        assert "_serialized" in override_models[0].__dict__

        models = [*tier_models.values(), *override_models]
        with ThreadPoolExecutor(max_workers=12) as pool:
            reads = list(pool.map(lambda model: model._serialized, models * 20))
        assert all(value is models[index % len(models)]._serialized
                   for index, value in enumerate(reads))
        assert graph.quick_thinking_llm is tier_models["quick"]
    finally:
        for name, factory in original_factories.items():
            setattr(graph_setup, name, factory)
        Reflector.__init__ = original_reflector_init
