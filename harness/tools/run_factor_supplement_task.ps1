[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,
    [Parameter(Mandatory = $true)]
    [string]$PythonExe,
    [string]$Config = "factor/research/factor_supplement_v1",
    [ValidateSet("dry-run", "execute")]
    [string]$Mode = "dry-run"
)

$ErrorActionPreference = "Stop"
$resolvedRepo = (Resolve-Path -LiteralPath $RepoRoot).Path
if (-not (Test-Path -LiteralPath (Join-Path $resolvedRepo "harness\tools\run_factor_supplement.py"))) {
    throw "research supplement CLI is missing under repo root: $resolvedRepo"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable does not exist: $PythonExe"
}

$scratchScript = @'
import os
from cbond_on.workflows.research.factor_supplement import configured_scratch_root, load_supplement_config
_path, cfg = load_supplement_config(os.environ['CBOND_ON_FACTOR_SUPPLEMENT_CONFIG'])
print(configured_scratch_root(cfg))
'@
$priorConfigEnv = $env:CBOND_ON_FACTOR_SUPPLEMENT_CONFIG
$env:CBOND_ON_FACTOR_SUPPLEMENT_CONFIG = $Config
Push-Location -LiteralPath $resolvedRepo
try {
    $scratchRoot = (& $PythonExe -c $scratchScript).Trim()
} finally {
    Pop-Location
    if ($null -eq $priorConfigEnv) {
        Remove-Item Env:CBOND_ON_FACTOR_SUPPLEMENT_CONFIG -ErrorAction SilentlyContinue
    } else {
        $env:CBOND_ON_FACTOR_SUPPLEMENT_CONFIG = $priorConfigEnv
    }
}
if (-not $scratchRoot) {
    throw "unable to resolve research scratch root from supplement configuration"
}
$timestamp = Get-Date -Format "yyyyMMddTHHmmss"
$logDir = Join-Path $scratchRoot "task_logs\$((Get-Date).ToString('yyyy-MM-dd'))"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stdoutPath = Join-Path $logDir "${timestamp}_${Mode}.stdout.log"
$stderrPath = Join-Path $logDir "${timestamp}_${Mode}.stderr.log"

Push-Location -LiteralPath $resolvedRepo
try {
    & $PythonExe -B "harness/tools/run_factor_supplement.py" --config $Config --mode $Mode 1> $stdoutPath 2> $stderrPath
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
[pscustomobject]@{
    Mode = $Mode
    ExitCode = $exitCode
    StdoutPath = $stdoutPath
    StderrPath = $stderrPath
} | ConvertTo-Json -Compress
exit $exitCode
