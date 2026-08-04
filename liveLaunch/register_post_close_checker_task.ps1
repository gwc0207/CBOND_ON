[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$TaskName = "CBOND_ON_PostCloseReadiness",
    [string]$At = "",
    [string]$RepoRoot = "",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
if (-not $RepoRoot) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}
$resolvedRepo = (Resolve-Path -LiteralPath $RepoRoot).Path
if (-not (Test-Path -LiteralPath (Join-Path $resolvedRepo "liveLaunch\post_close_checker.py"))) {
    throw "post-close checker is missing under repo root: $resolvedRepo"
}

if (-not $PythonExe) {
    $PythonExe = (& py -3 -c "import sys; print(sys.executable)").Trim()
}
if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) {
    throw "unable to resolve a Python executable; pass -PythonExe explicitly"
}

$configTimeScript = @'
from cbond_on.core.config import load_config_file
cfg = dict(load_config_file('live').get('post_close_readiness', {}))
print(str(cfg.get('check_time', '')).strip())
'@
Push-Location -LiteralPath $resolvedRepo
try {
    $configuredAt = (& $PythonExe -c $configTimeScript).Trim()
} finally {
    Pop-Location
}
if ($configuredAt -notmatch "^([01]\d|2[0-3]):[0-5]\d$") {
    throw "live.post_close_readiness.check_time must be HH:mm, got '$configuredAt'"
}
if ($At) {
    if ($At -notmatch "^([01]\d|2[0-3]):[0-5]\d$") {
        throw "-At must be HH:mm, got '$At'"
    }
    if ($At -ne $configuredAt) {
        throw "-At '$At' conflicts with live.post_close_readiness.check_time '$configuredAt'"
    }
} else {
    $At = $configuredAt
}

$action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "-m liveLaunch.post_close_checker" `
    -WorkingDirectory $resolvedRepo
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

if ($PSCmdlet.ShouldProcess("Task Scheduler/$TaskName", "register independent read-only post-close checker at $At")) {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Description "CBOND_ON independent configured-time read-only incident readiness check; never starts live scheduler or writes trade DB. Missed runs are not started late." `
        -Force | Out-Null
}

$registered = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($registered) {
    $registered | Select-Object TaskName, TaskPath, State
} elseif (-not $WhatIfPreference) {
    throw "Task Scheduler did not return the registered task: $TaskName"
}
