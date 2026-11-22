"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
)
from tradingagents.agents.schemas import (
    InstrumentHypothesis,
    TraderProposal,
    render_instrument_hypothesis,
    render_trader_proposal,
)
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm, research_only: bool = False):
    schema = InstrumentHypothesis if research_only else TraderProposal
    renderer = render_instrument_hypothesis if research_only else render_trader_proposal
    structured_llm = bind_structured(llm, schema, "Trader")

    def trader_node(state):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]
        # The research plan digests the debate but loses exact price structure;
        # give the Trader the technical market report so entry/stop levels are
        # grounded in real ATR / support-resistance / current price (#1167). The
        # report is empty when the user did not select the market analyst, so
        # only offer it (and the grounding instruction) when it has content.
        market_report = (state["market_report"] or "").strip()
        portfolio_context = get_portfolio_context_from_state(state)

        if research_only:
            grounding = "Use instrument prices only to describe possible market conditions; do not recommend a user-specific entry, order, share count, stop or allocation. "
            report_section = f"Technical Market Report:\n{market_report}\n\n" if market_report else ""
        elif market_report:
            grounding = (
                "Ground concrete price levels (entry, stop-loss, position sizing) in the technical "
                "market report's price structure -- current price, support/resistance, ATR, and "
                "volatility -- and use the research plan for direction and strategy. "
            )
            report_section = f"Technical Market Report:\n{market_report}\n\n"
        else:
            grounding = ""
            report_section = ""

        if research_only:
            messages = [
                {"role": "system", "content": (
                    "You are an instrument hypothesis reviewer. You do not know the user's holdings, cash, cost basis or portfolio. "
                    "Assess the instrument thesis and conditions that could improve or invalidate it. Never recommend buying, selling, holding a user's position, shares, cash fractions, portfolio allocation, standard allocation, entry orders or stops. "
                    + grounding + NO_EXTERNAL_TOOLS + get_language_instruction()
                )},
                {"role": "user", "content": (
                    f"Review the research thesis for {company_name}.\n{instrument_context}\n\n{report_section}"
                    f"Research plan:\n{investment_plan}\n\n"
                    "Return an instrument outlook, evidence-based reasoning, possible market conditions and thesis invalidation conditions. "
                    "Discuss absolute valuation or market levels only as uncertain instrument research, never an order."
                )},
            ]
        else:
            messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    + grounding
                    # Entry/stop are numeric price fields. Asking for concrete
                    # levels invites a percentage ("15%"), which is not a price
                    # and fails the structured parse (#1288).
                    + "State entry price and stop-loss as absolute price levels in the "
                    "instrument's quote currency (for example 189.5), never a percentage "
                    "or a range; convert a percentage distance to the price level it "
                    "implies, or omit the field if you cannot state a number. "
                    + NO_EXTERNAL_TOOLS
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the research team's investment plan for {company_name}. "
                    f"{instrument_context}\n\n"
                    f"{report_section}"
                    f"{portfolio_context}\n\n"
                    f"Proposed Investment Plan:\n{investment_plan}\n\n"
                    "Make an informed, strategic trading decision.\n\n"
                    "## Output\n\n"
                    "Write these sections, in this order, starting with the action "
                    "on its own line:\n\n"
                    "- **Action**: exactly one of Buy / Hold / Sell. A research "
                    "recommendation of Overweight is a Buy and Underweight is a Sell, "
                    "sized by how strong the case is; conflict alone is not a Hold.\n"
                    "- **Reasoning**: why, against the plan and the price structure\n"
                    "- **Entry Price**, **Stop Loss**, **Position Sizing**: when you can state them"
                ),
            },
            ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            renderer,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
        }

    return trader_node
