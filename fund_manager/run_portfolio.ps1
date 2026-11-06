<#
.SYNOPSIS
  Fund Manager add-on for TradingAgents. Flow: (optional) native per-ticker agents -> Fund Manager DRAFT
  -> portfolio risk committee debate (Aggressive / Conservative / Neutral) -> Fund Manager FINAL.
  Does not touch the project's own files.

.EXAMPLES
  .\addons\fund_manager\run_portfolio.ps1                        # latest reports, full flow (5 LLM calls)
  .\addons\fund_manager\run_portfolio.ps1 -RiskRounds 2          # two debate rounds (8 LLM calls)
  .\addons\fund_manager\run_portfolio.ps1 -RiskRounds 0          # draft only (1 call)
  .\addons\fund_manager\run_portfolio.ps1 -RunAgents             # rerun the 13 holdings first
  .\addons\fund_manager\run_portfolio.ps1 -RunAgents -IncludeWatchlist   # + agent reports for candidates
  .\addons\fund_manager\run_portfolio.ps1 -NoLLM                 # free quant snapshot only
#>
[CmdletBinding()]
param(
  [switch]$RunAgents,                      # run native TradingAgents per holding first
  [switch]$IncludeWatchlist,               # also run native TradingAgents on data\watchlist.json candidates
  [switch]$NoWatchlist,                    # ignore the watchlist entirely in the fund manager step
  [string]$AnalysisDate = "",              # --date for the native runs (default: previous weekday)
  [string]$Tickers = "",                   # comma list to limit native runs (default: all)
  [string]$Analysts = "market,sentiment,news,fundamentals",
  [string]$Model = "openai/gpt-6.1-sol",                   # fund manager (draft + final)
  [string]$AggressiveModel = "openai/gpt-6.1-sol",
  [string]$NeutralModel = "openai/gpt-6.1-sol",
  [string]$ConservativeModel = "anthropic/claude-opus-5.5",
  [int]$RiskRounds = 1,                    # 0 = no debate (draft is final)
  [string]$ReasoningEffort = "high",       # fund manager: low | medium | high | none
  [string]$AnalystEffort = "medium",       # risk analysts
  [double]$CashMin = -1,                   # override cash band (default from data\targets.json, 5-35)
  [double]$CashMax = -1,
  [switch]$IncludeRisk,                    # also feed per-stock risk-debate notes (bigger prompt)
  [switch]$NoLLM,
  [switch]$NoMacro,
  [string]$Since = ""                      # only use reports generated on/after YYYY-MM-DD (UTC)
)

$AddonDir   = $PSScriptRoot
$ProjectDir = (Resolve-Path (Join-Path $AddonDir "..\..")).Path
Set-Location $ProjectDir
$DataDir = Join-Path $ProjectDir "data"
$Start = Get-Date

if (-not (Test-Path (Join-Path $DataDir "my_book.json"))) { Write-Error "data\my_book.json not found"; exit 1 }
if (-not (Test-Path (Join-Path $DataDir "targets.json"))) {
  Copy-Item (Join-Path $AddonDir "targets.json") (Join-Path $DataDir "targets.json")
  Write-Host "Created data\targets.json from the add-on template (guidance weights)."
}
if (-not (Test-Path (Join-Path $DataDir "watchlist.json"))) { Set-Content (Join-Path $DataDir "watchlist.json") "[]" }

if (-not $AnalysisDate) {
  $d = (Get-Date).AddDays(-1)
  while ($d.DayOfWeek -in 'Saturday','Sunday') { $d = $d.AddDays(-1) }
  $AnalysisDate = $d.ToString("yyyy-MM-dd")
}

function Invoke-NativeRun([string[]]$list, [string]$kind) {
  # Uses add-on wrapper so news_data = yfinance,alpha_vantage without editing project source.
  $i = 0
  foreach ($t in $list) {
    $i++
    Write-Host "[$kind $i/$($list.Count)] TradingAgents $t ($AnalysisDate) [news: yfinance,alpha_vantage] ..." -ForegroundColor Cyan
    docker compose run --rm -T --entrypoint python -v "${AddonDir}:/home/appuser/addon:ro" tradingagents `
      /home/appuser/addon/run_native.py --ticker $t --date $AnalysisDate --analysts $Analysts `
      --portfolio /home/appuser/.tradingagents/my_book.json --save --no-show
    if ($LASTEXITCODE -ne 0) { Write-Warning "$t run exited with code $LASTEXITCODE; continuing" }
  }
}

$only = if ($Tickers) { $Tickers.Split(",") | ForEach-Object { $_.Trim().ToUpper() } } else { @() }
if ($RunAgents) {
  $book = Get-Content (Join-Path $DataDir "my_book.json") -Raw | ConvertFrom-Json
  $list = @($book.positions.ticker)
  if ($only.Count) { $list = @($list | Where-Object { $only -contains $_ }) }
  Invoke-NativeRun $list "holding"
}
if ($IncludeWatchlist) {
  # Candidates are run with the same book: they are not in it, so the agents see no position.
  $wl = Get-Content (Join-Path $DataDir "watchlist.json") -Raw | ConvertFrom-Json
  if ($wl -isnot [array] -and $wl.candidates) { $wl = $wl.candidates }
  $held = @((Get-Content (Join-Path $DataDir "my_book.json") -Raw | ConvertFrom-Json).positions.ticker)
  $cands = @($wl | ForEach-Object { if ($_ -is [string]) { $_ } else { $_.ticker } } | Where-Object { $_ -and ($held -notcontains $_) })
  if ($only.Count) { $cands = @($cands | Where-Object { $only -contains $_ }) }
  Invoke-NativeRun $cands "candidate"
}
# After a full holdings rerun, only use today's reports (UTC date). Partial reruns keep older reports for the rest.
if ($RunAgents -and -not $Tickers -and -not $Since) { $Since = $Start.ToUniversalTime().ToString("yyyy-MM-dd") }

$fmArgs = @("/home/appuser/addon/fund_manager.py", "--date", (Get-Date).ToString("yyyy-MM-dd"),
            "--model", $Model, "--aggressive-model", $AggressiveModel, "--neutral-model", $NeutralModel,
            "--conservative-model", $ConservativeModel, "--risk-rounds", "$RiskRounds",
            "--reasoning-effort", $ReasoningEffort, "--analyst-effort", $AnalystEffort)
if ($CashMin -ge 0)  { $fmArgs += @("--cash-min", "$CashMin") }
if ($CashMax -ge 0)  { $fmArgs += @("--cash-max", "$CashMax") }
if ($IncludeRisk)    { $fmArgs += "--include-risk" }
if ($NoLLM)          { $fmArgs += "--no-llm" }
if ($NoMacro)        { $fmArgs += "--no-macro" }
if ($NoWatchlist)    { $fmArgs += "--no-watchlist" }
if ($Since)          { $fmArgs += @("--since", $Since) }

Write-Host "Fund manager step ..." -ForegroundColor Cyan
docker compose run --rm -T --entrypoint python -v "${AddonDir}:/home/appuser/addon:ro" tradingagents @fmArgs
$code = $LASTEXITCODE
Write-Host ("Done in {0:N1} min. Reports: {1}" -f ((Get-Date) - $Start).TotalMinutes, (Join-Path $DataDir "fund_manager"))
exit $code
