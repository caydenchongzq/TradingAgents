"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call. Its rating is the run's
``final_rating``, and the decision is rendered to markdown as
``final_trade_decision`` for the memory log, CLI display and saved reports.
When a provider does not expose structured output, the agent falls back to
free-text generation and the rating is read from that text.
"""

from __future__ import annotations

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
)
from tradingagents.agents.rating import parse_rating
from tradingagents.agents.schemas import (
    InstrumentResearchBrief,
    PortfolioDecision,
    render_instrument_research_brief,
    render_pm_decision,
)
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, invoke_structured


def create_portfolio_manager(llm, research_only: bool = False):
    schema = InstrumentResearchBrief if research_only else PortfolioDecision
    renderer = render_instrument_research_brief if research_only else render_pm_decision
    structured_llm = bind_structured(llm, schema, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        portfolio_context = "" if research_only else get_portfolio_context_from_state(state)

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        if research_only:
            prompt = f"""You are the final instrument research synthesizer, not an allocator. You have no user holdings, cash, cost basis or portfolio. Do not recommend buying, selling, holding a user's position, share quantities, order timing, cash use, allocation percentages or standard allocations.

{instrument_context}

Synthesize the research manager's instrument thesis, the trader's hypothesis and the three instrument risk views below. Preserve disagreements and uncertainty. Provide a justified company valuation target or range only if the evidence supports it; otherwise leave the value unknown and explain why. State a thesis horizon only when supported. A thesis horizon describes the expected evidence window, not a user's holding period.

Research plan: {research_plan}
Instrument hypothesis: {trader_plan}
Risk debate:
{history}

Return the structured instrument research brief. Include thesis/counter-thesis, valuation method and assumptions, dated sentiment, catalysts, risks, conditional instrument levels, confidence, unknowns and source dates. Numeric financial facts and conditional price levels are instrument research, not portfolio sizing.
{NO_EXTERNAL_TOOLS}{get_language_instruction()}"""
        else:
            prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

{portfolio_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Ground every conclusion in specific evidence from the analysts. The risk debate always contains conflicting stances; deciding which is stronger is the job, so conflict alone is not a reason to Hold. Commit to the stronger case, sized by how decisively it wins. Choose Hold only when the evidence is still balanced after that weighing, or too thin to support a call; do not force a direction to appear decisive. Weigh the analysts on their merits, independent of speaking order.

## Output

Write these sections, in this order, starting with the rating on its own line:

- **Rating**: exactly one of Buy / Overweight / Hold / Underweight / Sell
- **Executive Summary**: the call and how to act on it
- **Investment Thesis**: the evidence that decided it, and what would change it

{NO_EXTERNAL_TOOLS}{get_language_instruction()}"""

        # The typed rating is the decision; the rendered text only carries it.
        # Read back from text, a rating the thesis quotes could replace it.
        decision = invoke_structured(structured_llm, prompt, "Portfolio Manager")
        if decision is not None:
            final_trade_decision = renderer(decision)
            final_rating = "RESEARCH" if research_only else decision.rating.value
        else:
            final_trade_decision = llm.invoke(prompt).content
            final_rating = "REVIEW" if research_only else parse_rating(final_trade_decision)

        new_risk_debate_state = {
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        result = {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
            "final_rating": final_rating,
        }
        if research_only and decision is not None:
            result["research_brief"] = decision.model_dump(mode="json")
        return result

    return portfolio_manager_node
