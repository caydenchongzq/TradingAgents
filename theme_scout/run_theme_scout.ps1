<#
.SYNOPSIS
  Weekly theme scout and free daily screen. Does not edit TradingAgents source.

.NOTES
  -Screen   ranked quant screen only (no model, no watchlist edit)
  -NoLLM    weekly macro/news/screen pack, no model, no watchlist edit
  -DryRun   same as -NoLLM
  A run with none of those flags makes one openai/gpt-6.1-sol call. In shadow mode (default until
  watchlist_curator.shadow_until) it only writes watchlist_proposals.json; after that it may update
  data\watchlist.json and append data\watchlist_log.jsonl. It stops first if the weekly budget is spent.

.EXAMPLES
  .\addons\theme_scout\run_theme_scout.ps1 -DryRun
  .\addons\theme_scout\run_theme_scout.ps1 -Screen
#>
[CmdletBinding()]
param(
  [switch]$Screen,
  [switch]$NoLLM,
  [switch]$DryRun,
  [string]$Date = "",
  [int]$TierN = 8,
  [string]$Model = "openai/gpt-6.1-sol",
  [switch]$KeepWatchlist                 # save the report; do not rewrite watchlist.json
)

$ErrorActionPreference = "Continue"
$ScoutDir = $PSScriptRoot
$ProjectDir = (Resolve-Path (Join-Path $ScoutDir "..\..")).Path
$FmDir = Join-Path $ProjectDir "addons\fund_manager"
Set-Location $ProjectDir
$NyTz = "EST5EDT,M3.2.0,M11.1.0"
$Addons = (Resolve-Path (Join-Path $ScoutDir "..")).Path
$pyArgs = @("/home/appuser/theme_scout/theme_scout.py", "--tier-n", "$TierN",
            "--themes", "/home/appuser/addon/themes.json")
if ($PSBoundParameters.ContainsKey("Model")) { $pyArgs += @("--model", $Model) }
if ($Date) { $pyArgs += @("--date", $Date) }
if ($Screen) { $pyArgs += "--screen" }
if ($NoLLM) { $pyArgs += "--no-llm" }
if ($DryRun) { $pyArgs += "--dry-run" }
if ($KeepWatchlist) { $pyArgs += "--keep-watchlist" }

if (-not ($Screen -or $NoLLM -or $DryRun)) {
  # Free check before the paid call. Same gate as run_portfolio.ps1.
  $ledger = Join-Path $Addons "cost_ledger.py"
  $dataDir = Join-Path $ProjectDir "data"
  $day = if ($Date) { $Date } else { (Get-Date).ToString("yyyy-MM-dd") }
  if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $ledger --data-root $dataDir --date $day --kind check --next 0.3
  } else {
    & python $ledger --data-root $dataDir --date $day --kind check --next 0.3
  }
  if ($LASTEXITCODE -eq 4) { Write-Error "Weekly budget would be exceeded. The scout call was not started."; exit 4 }
  if ($LASTEXITCODE -ne 0) { Write-Error "Weekly budget check failed (exit $LASTEXITCODE)."; exit $LASTEXITCODE }
}

& docker compose run --rm -T -e "TZ=$NyTz" --entrypoint python `
  -v "${ScoutDir}:/home/appuser/theme_scout:ro" `
  -v "${FmDir}:/home/appuser/addon:ro" `
  -v "${Addons}:/home/appuser/addons:ro" `
  tradingagents @pyArgs
exit $LASTEXITCODE
