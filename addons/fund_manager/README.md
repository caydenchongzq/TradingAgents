# Fund Manager add-on (TradingAgents)

A whole-portfolio "fund manager" with its own portfolio risk committee, on top of TradingAgents'
single-stock reports. The add-on lives in `addons/fund_manager/`; the account-blind research profile also extends
the installed native graph while preserving its ordinary portfolio-aware mode. Runs inside the existing
`tradingagents` Docker image (reuses yfinance, pandas, openai, the FRED vendor and `.env`).

## Flow
1. (`-RunAgents`) native TradingAgents per holding in `data\my_book.json`;
   (`-IncludeWatchlist`) also per candidate in `data\watchlist.json` (same book, so the agents see no position).
2. Latest report per ticker -> PM rating/decision, trader sizing, research manager, bull/bear and risk gists.
   Holdings honour `-Since`; candidates always use their latest report (missing reports are fine).
3. Quant layer (yfinance 2y daily + latest quote): weights, P&L, returns, SMA50/200, vol, beta 1y AND 2y,
   correlation matrix + clusters, portfolio vol/beta/VaR, risk contribution, stress (SPY -5/-10/-20% using the
   higher of 1y/2y beta, plus replay of the worst SPY 5-day window), concentration, drift, market regime
   (SPY trend, realized vol, breadth, VIX/rates from FRED, agent-rating mix), watchlist candidate stats
   (trend, vol, beta, correlation to the book, portfolio-vol change if added / swapped in).
4. **Fund Manager DRAFT** (Sol) - keep / trim / exit / add for every holding, swaps into candidates, dynamic cash target.
5. **Risk committee** debates the draft. Speaking order rotates by the decision date and the round, so neutral is not always last. Aggressive stays on Sol. Conservative is `anthropic/claude-opus-5.5`. Neutral is `openai/gpt-6-luna` and is kept different from the fund manager. `-AllowOpus` only lets an Opus model onto the fund manager, aggressive, or neutral seat.
6. **Fund Manager FINAL** (Sol) - reads the debate, issues the final plan, lists changes vs draft and how each debate point was treated.
7. The add-on validates advisory trades per account, applies the configured fee/minimum/residual rules, and reports accepted, adjusted, and rejected rows. It never submits orders.

## Accounts and trade sizing
`my_book.json` uses the shared account book contract. Each account keeps its own USD cash and US-stock positions;
the top-level `positions` list is a consolidated view for analysis. Equal tickers in different accounts remain
separate in the account detail. The fund manager sees account IDs and must name the account for each trade. Cash
cannot be pooled between accounts. Tiger or Moomoo imports are read-only snapshots managed by Argus; this add-on
does not connect to a broker or change TradingAgents' native portfolio API.

`targets.json` -> `trade_policy` defaults to `$3` per order, `$1,000` minimum buy notional, and a `$1,000`
residual-position guideline. The buy minimum applies per order and account; it is not a portfolio holding cap.
Buys and trims use whole shares. An exit can close the full sellable remainder, including fractional shares.
If a trim would leave a positive position below the residual guideline, validation adjusts it to a full available
exit. Valid small risk reductions are still allowed. Every accepted buy or sell incurs one fee. Unknown USD cash,
missing current prices, or unavailable sell quantity prevents validation. A known-cash order check is a simulation,
not an order preview or execution guarantee. Legacy `est_price` values never substitute for a current quote.
Rejected proposals do not change cash, quantities, fees, or risk summaries.

## Cash is dynamic
`data\targets.json` -> `cash_policy` `{min_pct: 5, max_pct: 35, soft_default_pct: 12}`. Each run the manager picks a
target cash % inside the band from the regime, macro, stress and opportunity set, justifies it and states deploy /
raise conditions. The risk team debates it. Override the band with `-CashMin` / `-CashMax`.

## Run (PowerShell)
```powershell
cd "D:\Software Development\trading\TradingAgents"
powershell -ExecutionPolicy Bypass -File .\addons\fund_manager\run_portfolio.ps1                       # full flow on latest reports (5 LLM calls)
powershell -ExecutionPolicy Bypass -File .\addons\fund_manager\run_portfolio.ps1 -RiskRounds 2         # 2 debate rounds (8 calls)
powershell -ExecutionPolicy Bypass -File .\addons\fund_manager\run_portfolio.ps1 -RiskRounds 0         # draft only (1 call)
powershell -ExecutionPolicy Bypass -File .\addons\fund_manager\run_portfolio.ps1 -RunAgents            # rerun 13 holdings first
powershell -ExecutionPolicy Bypass -File .\addons\fund_manager\run_portfolio.ps1 -IncludeWatchlist     # agent reports for candidates first
powershell -ExecutionPolicy Bypass -File .\addons\fund_manager\run_portfolio.ps1 -NoLLM                # free quant snapshot
```
Other options: `-ConservativeModel anthropic/claude-opus-5.5` (the default), `-NeutralModel openai/gpt-6-luna`, `-AggressiveModel`, `-Model` (fund manager), `-AllowOpus` (Opus on a seat other than conservative),
`-ReasoningEffort high` (manager), `-AnalystEffort medium`, `-Tickers NVDA,TSLA` (limit native runs), `-NoWatchlist`,
`-IncludeRisk` (feed per-stock risk notes), `-NoMacro`, `-Since YYYY-MM-DD`.

## Files
- `data\targets.json` - approved target weights, portfolio guidance, cash policy, trade policy, budget, and `ai_sleeve`.
  `data\watchlist.json` - candidates: a list of `{"ticker","sector","type","theme","note"}` (only ticker required) or plain tickers.
- `themes.json` (this folder) - theme/bucket tags for concentration. Names in `ai_growth` count fully in the AI sleeve unless `ai_sleeve.ticker_weight` sets another fraction.

## AI sleeve
`data\targets.json` -> `ai_sleeve`. `cap_pct` is the look-through warn level (50). `theme_weight` is the fraction of a theme that counts (ai_growth is 1). `ticker_weight` overrides that, and 0 excludes the ticker. Unlisted tickers keep the built-in defaults (MSFT, GOOG, and GOOGL at 1, SPY at 0.33, AAPL at 0). The add-on template is `addons\fund_manager\targets.json`.

AI look-through remains a separate risk summary based on `ai_sleeve` weights and general theme/bucket tags. The hardcoded AI subtheme taxonomy, subtheme fields, and subtheme report tables have been removed. Legacy `sub_theme` fields in a saved watchlist are ignored and are not emitted again.

AAPL is 0 because its Yahoo adjusted-close correlation to SMH, through the 2026-11-05 session, was about 0.05 over 252 sessions and -0.12 over 60 sessions. MSFT was 0.13 / 0.28 and GOOG was 0.28 / 0.16 on the same windows. The rule was to drop AAPL when that correlation is clearly below MSFT and GOOG (the example line was 0.5). On 5 Nov 2026, AAPL was -1.53% against SMH -11.21% and NVDA -9.26%.
- Prompts/schemas: `FM_SYSTEM`, `ANALYST_ROLES`, `ANALYST_TASK`, `*_SCHEMA` in `fund_manager.py`.

## Session date and prices
The run script sets the container timezone to `EST5EDT,M3.2.0,M11.1.0` (America/New_York, without tzdata) and takes the analysis date from that clock, so TradingAgents does not treat a live run as a backtest. Before anything else, `preflight.py` checks that each ticker's latest completed session bar has a real close, deletes a cache file that does not, and aborts if a refetch still has no close. `-RunAgents` also refuses to start while the US cash session is open, unless `-AllowIntraday`.

## Outputs: `data\fund_manager\<date>\<HHMMSS>_<id>\`
A `latest.json` pointer sits next to the date folders. Re-runs no longer overwrite the previous plan. `-NoLLM` still uses the `snapshot_` / `portfolio_snapshot` names inside its own folder.
- `portfolio_report.md` - cash decision, final plan, changes vs draft, debate, per-stock agent views side by side, quant appendix, LLM cost table
- `portfolio_report.json` - draft, debate, final, guardrail checks, stats, usage
- `portfolio_stats.json`, `prompt.md` - exact inputs, for auditing / refining
- `-NoLLM` writes `portfolio_snapshot.*` / `snapshot_*` instead, so it never overwrites a decision

## Instrument research profile and decision continuity

Selected native ticker refreshes use TradingAgents' existing graph with `--research-only`; the launch omits `--portfolio` and uses `--no-checkpoint`. The profile produces a versioned `research_brief.json` alongside the native reports. A justified valuation target/range and thesis horizon remain optional research fields; unknown values stay unknown. The profile runs only for tickers selected by the existing refresh scope, so weekly and earnings-driven refresh cadence remains in place. It does not start an automatic daily refresh or add another paid agent stage.

Research-only mode uses a separate `research_only_memory.md` path and skips reading, settling, or writing portfolio-aware native decision memory. Existing checkpoints are disabled for the run. Configured native seat models and effort remain in effect; saved custom native system-prompt replacements are suppressed for this profile to preserve its account-blind instructions. Fund-manager and theme-scout prompts are not changed by that native-only rule. The wrapper rejects `ARGUS_MACRO_CONTEXT_FILE` in research-only mode because it can contain whole-book scout context.

The whole-fund manager remains the sole portfolio allocator and receives all holdings plus the active watchlist universe, even when only selected tickers were refreshed. It consumes a validated versioned research brief when present. Its legacy fallback can use analyst, bull/bear and other evidence with provenance limits; legacy trader sizing and portfolio-manager/research-manager recommendations are not sizing authority. Coverage and dated/missing evidence are reported so limited refresh selection is visible.

Fund-manager decisions now carry proposed total shares, review horizon, sizing rationale, and current/deferred/none execution intent. The continuity section compares the selected prior completed review with account-local observed quantities and snapshot timestamps, while preserving unknown quantities and known-zero positions in complete snapshots. Prior-review ordering prefers precise UTC generation instants against the current-run boundary; New York report dates and run-directory order are fallbacks only when timestamps are unavailable. A deferred recommendation must include a future condition and has zero effective shares, fees, and cash movement in current validation. Current recommendations still pass through the existing per-account cash, price, fee, sellability and quantity checks. Hypothetical post-trade values and snapshot deltas are not fills; no broker path or automatic allocation/frequency rule was added.

The report contract is versioned. `reprocess_response.py` retains legacy and v2 schemas so saved responses can be recovered with the matching contract, without changing canonical reports or treating recovery output as a continuity baseline. The native and fund-manager artifacts remain in their existing report folders; fund-manager history stores `portfolio_report.json`, `portfolio_report.md`, `portfolio_stats.json`, and source references, while native research adds `research_brief.json` to that ticker's report directory.

Offline checks used during implementation include:

```bash
.venv/bin/python -m pytest -q tests/test_research_only_profile.py tests/test_structured_agent_prompts.py tests/test_structured_agents.py tests/test_reporting.py tests/test_portfolio_context.py tests/test_cli_commands.py tests/test_graph_end_to_end.py
.venv/bin/python addons/fund_manager/check_decision_contract.py
.venv/bin/python addons/fund_manager/check_trade_validation.py
.venv/bin/python addons/fund_manager/check_response_recovery.py
.venv/bin/python addons/fund_manager/check_offline.py
.venv/bin/python addons/check_run_native_research_guard.py
```

From the Argus checkout, relevant boundary/UI checks are `.venv/bin/python check_runner_offline.py`, `.venv/bin/python check_runner_budget_offline.py`, `.venv/bin/python check_offline.py`, and `node check_fund_manager_ui_offline.js`. These use fixtures only; no paid/live-model calls or live market/broker validation were performed. PowerShell runtime was unavailable, so Windows execution remains unverified. When running `addons/check_agent_config.py`, set `ARGUS_AGENT_CONFIG` to an isolated temporary config path so user-specific seat overrides cannot affect deterministic fixture expectations; do not rewrite the live config to satisfy the check. See the active Opmentix plan for current implementation, independent review, and acceptance evidence.
