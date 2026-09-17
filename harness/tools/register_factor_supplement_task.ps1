[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$TaskName = "CBOND_ON_FactorSupplementV1",
    [string]$RepoRoot = "",
    [string]$PythonExe = "",
    [string]$Config = "factor/research/factor_supplement_v1",
    [ValidateSet("dry-run", "execute")]
    [string]$Mode = "dry-run",
    [switch]$Register
)

$ErrorActionPreference = "Stop"
if (-not $RepoRoot) {
    $RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
}
$resolvedRepo = (Resolve-Path -LiteralPath $RepoRoot).Path
if (-not (Test-Path -LiteralPath (Join-Path $resolvedRepo "harness\tools\run_factor_supplement_task.ps1"))) {
    throw "research supplement task runner is missing under repo root: $resolvedRepo"
}

if (-not $PythonExe) {
    $PythonExe = (& py -3 -c "import sys; print(sys.executable)").Trim()
}
if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) {
    throw "unable to resolve a Python executable; pass -PythonExe explicitly"
}

$timeScript = @'
import os
from cbond_on.workflows.research.factor_supplement import configured_schedule_time, load_supplement_config
_path, cfg = load_supplement_config(os.environ['CBOND_ON_FACTOR_SUPPLEMENT_CONFIG'])
print(configured_schedule_time(cfg))
'@
$priorConfigEnv = $env:CBOND_ON_FACTOR_SUPPLEMENT_CONFIG
$env:CBOND_ON_FACTOR_SUPPLEMENT_CONFIG = $Config
Push-Location -LiteralPath $resolvedRepo
try {
    $configuredAt = (& $PythonExe -c $timeScript).Trim()
} finally {
    Pop-Location
    if ($null -eq $priorConfigEnv) {
        Remove-Item Env:CBOND_ON_FACTOR_SUPPLEMENT_CONFIG -ErrorAction SilentlyContinue
    } else {
        $env:CBOND_ON_FACTOR_SUPPLEMENT_CONFIG = $priorConfigEnv
    }
}
if ($configuredAt -ne "23:59") {
    throw "factor supplement V1 must remain scheduled at 23:59, got '$configuredAt'"
}

$taskRunner = Join-Path $resolvedRepo "harness\tools\run_factor_supplement_task.ps1"
$argument = "-NoProfile -ExecutionPolicy Bypass -File `"$taskRunner`" -RepoRoot `"$resolvedRepo`" -PythonExe `"$PythonExe`" -Config `"$Config`" -Mode $Mode"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argument -WorkingDirectory $resolvedRepo
$trigger = New-ScheduledTaskTrigger -Daily -At $configuredAt
$settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8) `
    -StartWhenAvailable:$false `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

if (-not $Register) {
    [pscustomobject]@{
        TaskName = $TaskName
        ScheduledAt = $configuredAt
        Mode = $Mode
        ActionExecute = "powershell.exe"
        ActionArguments = $argument
        StdoutPathPattern = "<scratch_root>\task_logs\YYYY-MM-DD\YYYYMMDDTHHMMSS_${Mode}.stdout.log"
        StderrPathPattern = "<scratch_root>\task_logs\YYYY-MM-DD\YYYYMMDDTHHMMSS_${Mode}.stderr.log"
        Registration = "not performed; rerun with -Register after owner confirmation"
    } | Format-List
    return
}

if ($PSCmdlet.ShouldProcess("Task Scheduler/$TaskName", "register independent research-only factor-supplement $Mode at $configuredAt")) {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Description "CBOND_ON research-only 23:59 factor-supplement V1 $Mode. It uses ephemeral scratch staging and publishes only the canonical factor-library table; it never invokes the live scheduler, DB, scores, or trade list." `
        -Force | Out-Null
}

$registered = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($registered) {
    $registered | Select-Object TaskName, TaskPath, State
} elseif (-not $WhatIfPreference) {
    throw "Task Scheduler did not return the registered task: $TaskName"
}
