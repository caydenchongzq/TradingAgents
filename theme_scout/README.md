# Theme scout

A separate add-on. It does not modify TradingAgents source.

## Weekly scout
One `openai/gpt-6.1-sol` call. Inputs are FRED (fed funds, 10y, 2y, curve, CPI, unemployment, VIX), a week of Yahoo and Alpha Vantage headlines, earnings dates, and 1-week / 1-month / 3-month relative strength of sector ETFs and AI sub-themes versus SPY. The model marks each sub-theme overweight, neutral, or avoid and proposes watchlist adds and drops. The list is kept between 30 and 50 names. Watchlist rules: 30–50 names, at most 5 adds and 5 drops a week, a 28-day cooldown after a name is added or dropped, holdings and staged orders are never dropped (flagged trim), new names count as AI weight 1.0 until tagged, and adds are blocked while the book is at the AI cap. Shadow mode is on until `watchlist_curator.shadow_until` in `data\targets.json` (2026-12-05): the scout writes `watchlist_proposals.json` in the run folder and does not edit the watchlist. After that, changes are written and appended to `data\watchlist_log.jsonl` (undo from the Argus Watchlist page). No backup file is written.

```powershell
cd "D:\Software Development\trading\TradingAgents"
powershell -ExecutionPolicy Bypass -File .\addons\theme_scout\run_theme_scout.ps1 -DryRun   # no model, no watchlist edit
powershell -ExecutionPolicy Bypass -File .\addons\theme_scout\run_theme_scout.ps1            # one Sol call, then the watchlist update
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
