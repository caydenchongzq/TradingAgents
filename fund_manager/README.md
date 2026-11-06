# Fund Manager add-on (TradingAgents)

A whole-portfolio "fund manager" with its own portfolio risk committee, on top of TradingAgents'
single-stock reports. Lives in `addons/fund_manager/`; the native project files are not modified.
Runs inside the existing `tradingagents` Docker image (reuses yfinance, pandas, openai, the FRED vendor and `.env`).

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
7. Guardrails recompute cash and post-trade weights, cap over-sells, flag cash outside the band or off-target.

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
- `data\targets.json` - guidance weights, cash policy, and `ai_sleeve`. `data\watchlist.json` - candidates: a list of
  `{"ticker","sector","type","theme","note"}` (only ticker required) or plain tickers.
- `themes.json` (this folder) - theme/bucket tags for concentration. Names in `ai_growth` count fully in the AI sleeve unless `ai_sleeve.ticker_weight` sets another fraction.

## AI sleeve
`data\targets.json` -> `ai_sleeve`. `theme_weight` is the fraction of a theme that counts (ai_growth is 1). `ticker_weight` overrides that, and 0 excludes the ticker. Unlisted tickers keep the built-in defaults (MSFT, GOOG, and GOOGL at 1, SPY at 0.33, AAPL at 0). The add-on template is `addons\fund_manager\targets.json`.

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
