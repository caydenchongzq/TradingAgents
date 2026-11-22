# Theme scout

A separate add-on. It does not modify TradingAgents source. It reads the shared account-book projection for portfolio context but does not connect to brokers or submit orders.

## Weekly scout
One `openai/gpt-6.1-sol` call. Inputs are FRED (fed funds, 10y, 2y, curve, CPI, unemployment, VIX), a week of Yahoo and Alpha Vantage headlines, earnings dates, and 1-week / 1-month / 3-month relative strength of selected sector ETFs versus SPY. The model marks general themes overweight, neutral, or avoid and proposes watchlist adds and drops. There is no hardcoded AI subtheme taxonomy; old `sub_theme` tags are ignored. The fixed cooldown is 14 days after a name is added or dropped; weekly add/drop caps are unlimited. Holdings and staged orders are protected from drops. Adds and drops are written as reviewable proposals and do not modify the live watchlist.

If any account has unknown USD cash or a held position has neither a current quote nor a usable cost, portfolio-wide AI exposure is unavailable. The scout continues to produce its free screen and reports, while leaving that whole-book metric blank.

```powershell
cd "D:\Software Development\trading\TradingAgents"
powershell -ExecutionPolicy Bypass -File .\addons\theme_scout\run_theme_scout.ps1 -DryRun   # no model, no watchlist edit
powershell -ExecutionPolicy Bypass -File .\addons\theme_scout\run_theme_scout.ps1            # one Sol call, then a reviewable proposal
```

Output: `data\theme_scout\<date>\report.md` and `report.json`.

## Daily screen
Free. Every holding and watchlist name is ranked on relative strength versus SPY, distance from the 20/50/200-day average, 14-day ATR, and earnings inside 7 days (a penalty).

```powershell
powershell -ExecutionPolicy Bypass -File .\addons\theme_scout\run_theme_scout.ps1 -Screen
```

Output: `data\theme_scout\<date>\screen.md` and `screen.json`. The stdout line `TIER_CANDIDATES=` is the top screened names that are not already holdings.

## Tiered portfolio run
`addons\fund_manager\run_portfolio.ps1 -Tiered` runs the free screen, then full TradingAgents on the holdings plus the top 8 candidates (`-TierN` changes the count). `-Tiered -NoLLM` prints that set and does not start agents.

Weekly cost is one Sol call. Recent Sol calls on this desk were about $0.15 to $0.29. A scout prompt is smaller than a full portfolio review, so a week is about $0.10 to $0.30. The daily screen is free.
