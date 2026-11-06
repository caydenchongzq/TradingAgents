<#
.SYNOPSIS
  Weekly theme scout and free daily screen. Does not edit TradingAgents source.

.NOTES
  -Screen   ranked quant screen only (no model, no watchlist edit)
  -NoLLM    weekly macro/news/screen pack, no model, no watchlist edit
  -DryRun   same as -NoLLM
  A run with none of those flags makes one openai/gpt-6.1-sol call and may update data\watchlist.json
  after writing a timestamped backup beside it.

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
  [string]$Model = "openai/gpt-6.1-sol"
)

$ErrorActionPreference = "Continue"
$ScoutDir = $PSScriptRoot
$ProjectDir = (Resolve-Path (Join-Path $ScoutDir "..\..")).Path
$FmDir = Join-Path $ProjectDir "addons\fund_manager"
Set-Location $ProjectDir
$NyTz = "EST5EDT,M3.2.0,M11.1.0"
$pyArgs = @("/home/appuser/theme_scout/theme_scout.py", "--tier-n", "$TierN", "--model", $Model,
            "--themes", "/home/appuser/addon/themes.json")
if ($Date) { $pyArgs += @("--date", $Date) }
if ($Screen) { $pyArgs += "--screen" }
if ($NoLLM) { $pyArgs += "--no-llm" }
if ($DryRun) { $pyArgs += "--dry-run" }

& docker compose run --rm -T -e "TZ=$NyTz" --entrypoint python `
  -v "${ScoutDir}:/home/appuser/theme_scout:ro" `
  -v "${FmDir}:/home/appuser/addon:ro" `
  tradingagents @pyArgs
exit $LASTEXITCODE
