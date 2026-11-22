"""Free checks for agent_config. No network and no LLM calls."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "fund_manager"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "theme_scout"))

import agent_config as ac
import fund_manager as fm
import theme_scout as ts


def check_managed_factory_forwarding() -> None:
    setup = types.ModuleType("tradingagents.graph.setup")

    def factory(llm, research_only=False, *, marker=None):
        return {"llm": llm, "research_only": research_only, "marker": marker}

    setup.create_market_analyst = factory
    reflection = types.ModuleType("tradingagents.memory.reflection")
    reflection.Reflector = type("Reflector", (), {"__init__": lambda self, llm: None})
    modules = {
        "tradingagents": types.ModuleType("tradingagents"),
        "tradingagents.graph": types.ModuleType("tradingagents.graph"),
        "tradingagents.graph.setup": setup,
        "tradingagents.memory": types.ModuleType("tradingagents.memory"),
        "tradingagents.memory.reflection": reflection,
    }
    custom_client = SimpleNamespace(model="configured-client")
    built = []
    with patch.dict(sys.modules, modules), patch.object(
        ac, "_NATIVE_FACTORIES", {"market": ("tradingagents.graph.setup", "create_market_analyst")}
    ), patch.object(ac, "_build_client", side_effect=lambda seat, config: built.append(seat["id"]) or custom_client):
        ac.install_native_seats({})
        managed = setup.create_market_analyst
        default_seat_model = ac.resolved_seat("market")["model"]

        # Ordinary and research calls both pass the native profile argument to
        # the real factory; the bool can no longer overwrite the captured _orig.
        ordinary = managed(SimpleNamespace(model=default_seat_model), False, marker="ordinary")
        research = managed(SimpleNamespace(model=default_seat_model), True, marker="research")
        assert isinstance(ordinary["llm"], ac._SeatCall)
        assert isinstance(research["llm"], ac._SeatCall)
        assert ordinary["research_only"] is False and ordinary["marker"] == "ordinary"
        assert research["research_only"] is True and research["marker"] == "research"

        # A seat-specific model override still builds/uses its own client.
        configured = managed(SimpleNamespace(model="different-model"), True, marker="own-client")
        assert configured["llm"].inner is custom_client
        assert configured["research_only"] is True and configured["marker"] == "own-client"
        assert built == ["market"]


def _run_checks() -> int:
    seats = {s["id"]: s for s in ac.resolved_seats()}
    assert len(seats) == 18, len(seats)
    for seat_id in (
        "theme_scout", "market", "sentiment", "news", "fundamentals", "bull", "bear",
        "research_manager", "trader", "risk_aggressive", "risk_conservative", "risk_neutral",
        "portfolio_manager", "memory", "fm_manager", "fm_aggressive", "fm_conservative", "fm_neutral",
    ):
        assert seat_id in seats, seat_id
        assert seats[seat_id]["system"].strip(), seat_id
    assert seats["fm_conservative"]["model"] == ac.LOCKED_MODEL
    assert seats["risk_conservative"]["model"] == ac.LOCKED_MODEL
    assert not seats["risk_conservative"]["locked"]
    assert not seats["fm_conservative"]["locked"]
    assert seats["fm_manager"]["est_cost_usd"] is not None
    assert seats["market"]["est_cost_usd"] is None
    assert ac.opus_warning("anthropic/claude-opus-5.5", "market")
    assert ac.opus_warning(ac.LOCKED_MODEL, "fm_conservative") is None
    assert ac._clean_model("gpt-4o-mini") == "gpt-4o-mini"
    assert ac._clean_model("meta/llama-3.1-8b-instruct:free") == "meta/llama-3.1-8b-instruct:free"
    assert ac.estimate_cost({"id": "fm_manager", "est_input_tokens": 120_438,
                             "est_output_tokens": 28_348}, {
        "usd_per_mtok_in": 2, "usd_per_mtok_out": 10,
        "price_tiers": [{"min_prompt_tokens": 60_000,
                         "usd_per_mtok_in": 4, "usd_per_mtok_out": 15}],
    }) is None, "combined draft/final samples cannot be accurately priced across tiers"
    assert "pricing tiers cannot be determined separately" in ac.resolved_seat(
        "fm_manager", {"models": [{
            "id": ac.resolved_seat("fm_manager")["model"],
            "usd_per_mtok_in": 2, "usd_per_mtok_out": 10,
            "price_tiers": [{"min_prompt_tokens": 60_000,
                             "usd_per_mtok_in": 4, "usd_per_mtok_out": 15}],
        }]}
    )["est_cost_note"]
    assert ac.model_row_has_conditional_pricing([{
        "id": seats["fm_manager"]["model"], "has_conditional_pricing": True,
    }], seats["fm_manager"]["model"])

    original = "prefix " + seats["bull"]["system"] + " suffix"
    assert ac.apply_system_prompt("bull", original) == original
    # A saved override replaces the anchor.
    original_agent_config = os.environ.get("ARGUS_AGENT_CONFIG")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "agent_config.json"
        os.environ["ARGUS_AGENT_CONFIG"] = str(path)
        try:
            path.write_text(json.dumps({"seats": {"market": {"model": "gpt-4o-mini"}}}))
            legacy_edit = ac.validate_and_save({"seats": {
                "market": {"model": "gpt-4o-mini", "system": "Edited legacy model prompt"},
            }})
            legacy_seats = {seat["id"]: seat for seat in legacy_edit["seats"]}
            assert legacy_seats["market"]["model"] == "gpt-4o-mini"
            assert legacy_seats["market"]["system"] == "Edited legacy model prompt"

            saved = ac.validate_and_save({
                "seats": {"fm_conservative": {
                    "model": "openai/gpt-6-luna",
                    "effort": "low",
                    "system": seats["fm_conservative"]["system"],
                }},
            })
            assert saved["seats"]
            got = {s["id"]: s for s in saved["seats"]}
            assert got["fm_conservative"]["model"] == "openai/gpt-6-luna"
            assert not got["fm_conservative"]["locked"]
            assert fm.agent_plan_from_config()["fm_conservative"]["model"] == "openai/gpt-6-luna"
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            assert "unlocked" not in on_disk
            assert "fm_conservative" in on_disk["seats"]
            assert on_disk["seats"]["fm_conservative"]["model"] == "openai/gpt-6-luna"
            assert list(path.parent.glob("*.bak")) == []
            free_variant = ac.validate_and_save({"seats": {"market": {
                "model": "meta/llama-3.1-8b-instruct:free",
                "system": seats["market"]["system"],
            }}})
            free_seats = {seat["id"]: seat for seat in free_variant["seats"]}
            assert free_seats["market"]["model"] == "meta/llama-3.1-8b-instruct:free"
            rewritten = ac.apply_system_prompt("bull", original)
            # bull was not overridden, so the prompt is unchanged
            assert rewritten == original
            saved_bull = ac.validate_and_save({
                "unlock": [],
                "seats": {
                    "bull": {"model": seats["bull"]["model"], "system": "CUSTOM BULL ROLE"},
                    "market": {"model": "anthropic/claude-opus-5.5", "system": seats["market"]["system"]},
                },
            })
            assert any("market" in w for w in saved_bull["warnings"])
            assert "CUSTOM BULL ROLE" in ac.apply_system_prompt("bull", original)
            ac.validate_and_save({
                "seats": {"fm_manager": {"model": seats["fm_manager"]["model"], "system": "CUSTOM FM ROLE"}},
            })
            fm_original = "prefix " + seats["fm_manager"]["system"] + " suffix"
            original_research_flag = os.environ.get("ARGUS_RESEARCH_ONLY")
            os.environ["ARGUS_RESEARCH_ONLY"] = "1"
            try:
                assert ac.apply_system_prompt("bull", original) == original
                assert "CUSTOM FM ROLE" in ac.apply_system_prompt("fm_manager", fm_original)
                assert ac.resolved_seat("bull")["model"] == got["bull"]["model"]
            finally:
                if original_research_flag is None:
                    os.environ.pop("ARGUS_RESEARCH_ONLY", None)
                else:
                    os.environ["ARGUS_RESEARCH_ONLY"] = original_research_flag
            ac.validate_and_save({
                "unlock": [],
                "seats": {"theme_scout": {
                    "model": seats["theme_scout"]["model"],
                    "effort": None,
                    "max_tokens": None,
                    "system": seats["theme_scout"]["system"],
                }},
            })
            scout_seat = ac.resolved_seat("theme_scout")
            assert scout_seat["effort"] is None
            assert scout_seat["max_tokens"] == 6000
            disk = json.loads(path.read_text(encoding="utf-8"))
            assert disk["seats"]["theme_scout"]["effort"] is None
            assert "max_tokens" not in disk["seats"]["theme_scout"]
            assert ts.agent_plan_from_config()["theme_scout"]["effort"] is None
        finally:
            if original_agent_config is None:
                os.environ.pop("ARGUS_AGENT_CONFIG", None)
            else:
                os.environ["ARGUS_AGENT_CONFIG"] = original_agent_config

    plan = fm.agent_plan_from_config()
    assert plan["fm_manager"]["model"] == "openai/gpt-6.1-sol", plan["fm_manager"]
    assert plan["fm_conservative"]["model"] == ac.LOCKED_MODEL
    assert plan["fm_neutral"]["model"] == "openai/gpt-6-luna"
    assert "Fund Manager" in plan["fm_manager"]["system"]
    scout = ts.agent_plan_from_config()["theme_scout"]
    assert scout["model"] == "openai/gpt-6.1-sol"
    assert "theme scout" in scout["system"]
    text = ac.dry_run_text()
    assert "no LLM calls" in text and "fm_conservative" in text
    pipe = ac.check_seat_pipe()
    assert pipe in ("ok", "skipped"), pipe
    check_managed_factory_forwarding()
    print("check_agent_config: ok")
    print(f"  seat pipe {pipe}")
    print(f"  seats {len(seats)} fm cost {seats['fm_manager']['est_cost_usd']}")
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="agent_config_check_") as tmp:
        fixture = {
            "ARGUS_AGENT_CONFIG": str(Path(tmp) / "agent_config.json"),
            "ARGUS_MODELS_CONFIG": str(Path(__file__).with_name("openrouter_models.json").resolve()),
        }
        with patch.dict(os.environ, fixture):
            return _run_checks()


if __name__ == "__main__":
    sys.exit(main())
