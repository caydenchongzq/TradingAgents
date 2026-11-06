<#
.SYNOPSIS
  Fund Manager add-on for TradingAgents. Flow: preflight (session date + settled bars)
  -> (optional) native per-ticker agents -> Fund Manager DRAFT
  -> portfolio risk committee debate (Aggressive / Conservative / Neutral) -> Fund Manager FINAL.
  Does not touch the project's own files.

.NOTES
  Every container is started with TZ=EST5EDT,M3.2.0,M11.1.0, the POSIX form of
  America/New_York. python:3.13-slim has no tzdata, so the IANA name would not
  move date.today(). The analysis date defaults to that New York calendar date,
  which is what TradingAgents compares in is_historical(). A past -AnalysisDate
  is still a historical run.

  Conservative and neutral default to openai/gpt-6-luna (the quick model already
  used for per-stock runs). Opus is not a default. -AllowOpus passes --allow-opus.

.EXAMPLES
  .\addons\fund_manager\run_portfolio.ps1                        # latest reports, full flow (5 LLM calls)
  .\addons\fund_manager\run_portfolio.ps1 -RiskRounds 2          # two debate rounds (8 LLM calls)
  .\addons\fund_manager\run_portfolio.ps1 -RiskRounds 0          # draft only (1 call)
  .\addons\fund_manager\run_portfolio.ps1 -RunAgents             # rerun the holdings first (after the NY close)
  .\addons\fund_manager\run_portfolio.ps1 -RunAgents -IncludeWatchlist
  .\addons\fund_manager\run_portfolio.ps1 -NoLLM                 # free quant snapshot only
#>
[CmdletBinding()]
param(
  [switch]$RunAgents,                      # run native TradingAgents per holding first
  [switch]$IncludeWatchlist,               # also run native TradingAgents on data\watchlist.json candidates
  [switch]$NoWatchlist,                    # ignore the watchlist entirely in the fund manager step
  [string]$AnalysisDate = "",              # blank = New York calendar date (live mode)
  [string]$Tickers = "",                   # comma list to limit native runs (default: all)
  [string]$Analysts = "market,sentiment,news,fundamentals",
  [string]$Model = "openai/gpt-6.1-sol",                   # fund manager (draft + final)
  [string]$AggressiveModel = "openai/gpt-6.1-sol",
  [string]$NeutralModel = "openai/gpt-6-luna",             # different model from the fund manager
  [string]$ConservativeModel = "openai/gpt-6-luna",        # was Opus; Luna is the cheap model already in use
  [switch]$AllowOpus,                      # pass an Opus model through instead of replacing it
  [int]$RiskRounds = 1,                    # 0 = no debate (draft is final)
  [string]$ReasoningEffort = "high",       # fund manager: low | medium | high | none
  [string]$AnalystEffort = "medium",       # risk analysts
  [double]$CashMin = -1,                   # override cash band (default from data\targets.json, 5-35)
  [double]$CashMax = -1,
  [switch]$IncludeRisk,                    # also feed per-stock risk-debate notes (bigger prompt)
  [switch]$NoLLM,
  [switch]$NoMacro,
  [switch]$AllowIntraday,                  # live agent run while the US cash session is open
  [string]$Since = ""                      # only use reports generated on/after YYYY-MM-DD (UTC)
)

$ErrorActionPreference = "Continue"
$AddonDir   = $PSScriptRoot
$ProjectDir = (Resolve-Path (Join-Path $AddonDir "..\..")).Path
Set-Location $ProjectDir
$DataDir = Join-Path $ProjectDir "data"
$Start = Get-Date
# POSIX America/New_York. Verified in this image: date.today() follows the New York calendar
# and the clock is EST/EDT, five hours behind UTC. Do not switch this to America/New_York
# unless the image has tzdata.
$NyTz = "EST5EDT,M3.2.0,M11.1.0"

if (-not (Test-Path (Join-Path $DataDir "my_book.json"))) { Write-Error "data\my_book.json not found"; exit 1 }
if (-not (Test-Path (Join-Path $DataDir "targets.json"))) {
  Copy-Item (Join-Path $AddonDir "targets.json") (Join-Path $DataDir "targets.json")
  Write-Host "Created data\targets.json from the add-on template (guidance weights)."
}
if (-not (Test-Path (Join-Path $DataDir "watchlist.json"))) { Set-Content (Join-Path $DataDir "watchlist.json") "[]" }

function Invoke-TaPython([string[]]$PyArgs) {
  & docker compose run --rm -T -e "TZ=$NyTz" --entrypoint python -v "${AddonDir}:/home/appuser/addon:ro" tradingagents @PyArgs
}

function Get-BookTickers {
  $book = Get-Content (Join-Path $DataDir "my_book.json") -Raw | ConvertFrom-Json
  return @($book.positions.ticker | ForEach-Object { "$_".Trim().ToUpper() } | Where-Object { $_ })
}

function Get-WatchTickers {
  $wl = Get-Content (Join-Path $DataDir "watchlist.json") -Raw | ConvertFrom-Json
  if ($wl -isnot [array] -and $wl.candidates) { $wl = $wl.candidates }
  return @($wl | ForEach-Object { if ($_ -is [string]) { $_ } else { $_.ticker } } | ForEach-Object { "$_".Trim().ToUpper() } | Where-Object { $_ })
}

$only = if ($Tickers) { @($Tickers.Split(",") | ForEach-Object { $_.Trim().ToUpper() } | Where-Object { $_ }) } else { @() }
$check = @(Get-BookTickers)
if (-not $NoWatchlist) { $check += @(Get-WatchTickers) }
if ($only.Count) { $check = @($check | Where-Object { $only -contains $_ }) }
$check = @($check | Select-Object -Unique)
if (-not $check.Count) { Write-Error "No tickers to preflight."; exit 2 }

$pfArgs = @("/home/appuser/addon/preflight.py", "--tickers", ($check -join ","))
if ($AnalysisDate) { $pfArgs += @("--analysis-date", $AnalysisDate) }
if ($AllowIntraday) { $pfArgs += "--allow-intraday" }
if ($RunAgents) { $pfArgs += "--block-intraday" }
Write-Host "Preflight (New York session, settled closes) ..." -ForegroundColor Cyan
$pfRaw = & docker compose run --rm -T -e "TZ=$NyTz" --entrypoint python -v "${AddonDir}:/home/appuser/addon:ro" tradingagents @pfArgs 2>&1
$pfCode = $LASTEXITCODE
$pfRaw | ForEach-Object { Write-Host $_ }
$pfText = ($pfRaw | Out-String)
if ($pfText -match 'ANALYSIS_DATE=(\d{4}-\d{2}-\d{2})') { $AnalysisDate = $Matches[1] }
$status = if ($pfText -match 'PREFLIGHT_STATUS=(\S+)') { $Matches[1] } else { "" }
if ($pfCode -ne 0 -or $status -ne "ok") {
  Write-Error "Preflight aborted (status='$status', exit=$pfCode). No agent or fund-manager run was started. If a cache file had no real close it was deleted so the next attempt refetches it."
  if ($pfCode -ne 0) { exit $pfCode }
  exit 3
}
Write-Host "Analysis date $AnalysisDate (New York). Container TZ is $NyTz." -ForegroundColor Green

function Invoke-NativeRun([string[]]$list, [string]$kind) {
  # Uses add-on wrapper so news_data = yfinance,alpha_vantage without editing project source.
  $failed = @()
  $i = 0
  foreach ($t in $list) {
    $i++
    Write-Host "[$kind $i/$($list.Count)] TradingAgents $t ($AnalysisDate) [news: yfinance,alpha_vantage] ..." -ForegroundColor Cyan
    Invoke-TaPython @("/home/appuser/addon/run_native.py", "--ticker", $t, "--date", $AnalysisDate, "--analysts", $Analysts,
      "--portfolio", "/home/appuser/.tradingagents/my_book.json", "--save", "--no-show") | Out-Host
    if ($LASTEXITCODE -ne 0) {
      Write-Warning "$t run exited with code $LASTEXITCODE"
      $failed += $t
    }
  }
  return $failed
}

$nativeFailed = @()
if ($RunAgents) {
  $list = @(Get-BookTickers)
  if ($only.Count) { $list = @($list | Where-Object { $only -contains $_ }) }
  $nativeFailed += @(Invoke-NativeRun $list "holding")
}
if ($IncludeWatchlist) {
  $held = @(Get-BookTickers)
  $cands = @(Get-WatchTickers | Where-Object { $held -notcontains $_ })
  if ($only.Count) { $cands = @($cands | Where-Object { $only -contains $_ }) }
  $nativeFailed += @(Invoke-NativeRun $cands "candidate")
}
# After a full holdings rerun, only use this session's reports. Partial reruns keep older reports for the rest.
if ($RunAgents -and -not $Tickers -and -not $Since) { $Since = $AnalysisDate }

$fmArgs = @("/home/appuser/addon/fund_manager.py", "--date", $AnalysisDate,
            "--model", $Model, "--aggressive-model", $AggressiveModel, "--neutral-model", $NeutralModel,
            "--conservative-model", $ConservativeModel, "--risk-rounds", "$RiskRounds",
            "--reasoning-effort", $ReasoningEffort, "--analyst-effort", $AnalystEffort)
if ($AllowOpus)      { $fmArgs += "--allow-opus" }
if ($CashMin -ge 0)  { $fmArgs += @("--cash-min", "$CashMin") }
if ($CashMax -ge 0)  { $fmArgs += @("--cash-max", "$CashMax") }
if ($IncludeRisk)    { $fmArgs += "--include-risk" }
if ($NoLLM)          { $fmArgs += "--no-llm" }
if ($NoMacro)        { $fmArgs += "--no-macro" }
if ($NoWatchlist)    { $fmArgs += "--no-watchlist" }
if ($Since)          { $fmArgs += @("--since", $Since) }

Write-Host "Fund manager step ..." -ForegroundColor Cyan
Invoke-TaPython $fmArgs | Out-Host
$code = $LASTEXITCODE
if ($nativeFailed.Count) {
  Write-Warning ("Native runs failed: " + ($nativeFailed -join ", ") + ". The fund manager still ran; those names may be missing a fresh report.")
  if ($code -eq 0) { $code = 1 }
}
Write-Host ("Done in {0:N1} min. Reports: {1}" -f ((Get-Date) - $Start).TotalMinutes, (Join-Path $DataDir "fund_manager"))
exit $code
