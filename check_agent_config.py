"""Free checks for agent_config. No network and no LLM calls."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "fund_manager"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "theme_scout"))

import agent_config as ac
import fund_manager as fm
import theme_scout as ts


def main() -> int:
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
    assert seats["risk_conservative"]["locked"] and not seats["risk_conservative"]["unlocked"]
    assert seats["fm_manager"]["est_cost_usd"] is not None
    assert seats["market"]["est_cost_usd"] is None
    assert ac.opus_warning("anthropic/claude-opus-5.5", "market")
    assert ac.opus_warning(ac.LOCKED_MODEL, "fm_conservative") is None

    original = "prefix " + seats["bull"]["system"] + " suffix"
    assert ac.apply_system_prompt("bull", original) == original
    # A saved override replaces the anchor.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "agent_config.json"
        os.environ["ARGUS_AGENT_CONFIG"] = str(path)
        try:
            try:
                ac.validate_and_save({
                    "unlock": [],
                    "seats": {"fm_conservative": {"model": "openai/gpt-6-luna", "system": seats["fm_conservative"]["system"]}},
                })
                raise AssertionError("locked seat accepted a new model")
            except ac.AgentConfigError:
                pass
            saved = ac.validate_and_save({
                "unlock": ["fm_conservative"],
                "seats": {"fm_conservative": {
                    "model": "openai/gpt-6-luna",
                    "effort": "low",
                    "system": seats["fm_conservative"]["system"],
                }},
            })
            assert saved["seats"]
            got = {s["id"]: s for s in saved["seats"]}
            assert got["fm_conservative"]["model"] == "openai/gpt-6-luna"
            assert got["fm_conservative"]["opus_warning"] is None
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            assert "fm_conservative" in on_disk["seats"]
            assert list(path.parent.glob("*.bak")) == []
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
            os.environ.pop("ARGUS_AGENT_CONFIG", None)

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
    print("check_agent_config: ok")
    print(f"  seats {len(seats)} fm cost {seats['fm_manager']['est_cost_usd']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
