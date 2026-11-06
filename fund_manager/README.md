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
5. **Risk committee** debates the draft: Aggressive (Sol) -> Conservative (Opus 5.5) -> Neutral (Sol), `-RiskRounds` rounds.
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
Other options: `-ConservativeModel anthropic/claude-opus-5.5`, `-AggressiveModel`, `-NeutralModel`, `-Model` (fund manager),
`-ReasoningEffort high` (manager), `-AnalystEffort medium`, `-Tickers NVDA,TSLA` (limit native runs), `-NoWatchlist`,
`-IncludeRisk` (feed per-stock risk notes), `-NoMacro`, `-Since YYYY-MM-DD`.

## Files
- `data\targets.json` - guidance weights + cash policy. `data\watchlist.json` - candidates: a list of
  `{"ticker","sector","type","theme","note"}` (only ticker required) or plain tickers.
- `themes.json` (this folder) - theme/bucket tags for concentration.
- Prompts/schemas: `FM_SYSTEM`, `ANALYST_ROLES`, `ANALYST_TASK`, `*_SCHEMA` in `fund_manager.py`.

## Outputs: `data\fund_manager\<date>\`
- `portfolio_report.md` - cash decision, final plan, changes vs draft, debate, per-stock agent views side by side, quant appendix, LLM cost table
- `portfolio_report.json` - draft, debate, final, guardrail checks, stats, usage
- `portfolio_stats.json`, `prompt.md` - exact inputs, for auditing / refining
- `-NoLLM` writes `portfolio_snapshot.*` / `snapshot_*` instead, so it never overwrites a decision
