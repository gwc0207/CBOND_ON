[CmdletBinding()]
param(
    [string]$QueueRoot = "D:\cbond_on\research_scratch\factor_mining_20260803_to50_serial_queue_r1",
    [int]$PollSeconds = 120,
    [double]$MinFreeGb = 6.0,
    [switch]$PlanOnly
)

<#
Serial, fail-closed research-only queue for the factor-mining 50-candidate
target.  It never touches a live config, production FactorStore, DB, model
state, scheduler, or output.  The first expansion must finish successfully
before the v8 batch is started; source drift blocks the queued work.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$ScratchParent = "D:\cbond_on\research_scratch"
$Py = "py"

$ActiveV5Root = "D:\cbond_on\research_scratch\factor_mining_20260803_aggregate_catalog_v5_full"
$ActiveV5Pid = 1572
$V8Root = "D:\cbond_on\research_scratch\factor_mining_20260803_daily_orthogonal_batch_v8_full_r1"
$V7MergedRoot = "D:\cbond_on\research_scratch\factor_mining_20260803_unified_v7_with_joint_merged_v1"
$V7Catalog = "D:\cbond_on\research_scratch\factor_mining_20260803_unified_v7_with_joint_catalog_v1\family_catalog.json"
$CatalogRoot = "D:\cbond_on\research_scratch\factor_mining_20260803_unified_v9_v5_v8_catalog_r1"
$MergedRoot = "D:\cbond_on\research_scratch\factor_mining_20260803_unified_v9_v5_v8_merged_r1"
$ScreenRoot = "D:\cbond_on\research_scratch\factor_mining_20260803_unified_v9_v5_v8_screen_r1"
$SelectionRoot = "D:\cbond_on\research_scratch\factor_mining_20260803_unified_v9_v5_v8_selection_optimal_r1"

$Runner = Join-Path $RepoRoot "harness\tools\run_factor_mining_expansion.py"
$Composer = Join-Path $RepoRoot "harness\tools\compose_research_factor_catalog.py"
$Merger = Join-Path $RepoRoot "harness\tools\merge_factor_mining_stores.py"
$Screen = Join-Path $RepoRoot "harness\tools\factor_mining_screen.py"
$Optimizer = Join-Path $RepoRoot "harness\tools\optimize_factor_mining_selection.py"
$FactorConfig = Join-Path $RepoRoot "cbond_on\config\factor\research\factor_mining_20260802_config.json5"
$PathsConfig = Join-Path $RepoRoot "cbond_on\config\data\paths_factor_mining_20260802_config.json5"
$FactorRuntime = Join-Path $RepoRoot "cbond_on\app\usecases\factor_batch_runtime.py"
$ResearchWorkflow = Join-Path $RepoRoot "cbond_on\workflows\research\factor_batch.py"
$ResearchBootstrap = Join-Path $RepoRoot "cbond_on\bootstrap\research.py"
$FactorBase = Join-Path $RepoRoot "cbond_on\domain\factors\base.py"
$ResearchDefsRoot = Join-Path $RepoRoot "cbond_on\domain\factors\defs"

function Assert-ScratchChild {
    param([Parameter(Mandatory = $true)][string]$Path)

    $resolved = [System.IO.Path]::GetFullPath($Path)
    $parent = [System.IO.Path]::GetFullPath($ScratchParent)
    if (-not $resolved.StartsWith($parent + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "research output escaped scratch parent: $resolved"
    }
}

function Assert-Absent {
    param([Parameter(Mandatory = $true)][string]$Path)

    Assert-ScratchChild -Path $Path
    if (Test-Path -LiteralPath $Path) {
        throw "refusing to reuse existing output root: $Path"
    }
}

function Write-Status {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [hashtable]$Extra = @{}
    )

    $payload = [ordered]@{
        schema_version = 1
        updated_at = (Get-Date).ToString("o")
        research_only = $true
        target_exact_mis_count = 50
        stage = $Stage
        active_v5_root = $ActiveV5Root
        v8_root = $V8Root
        catalog_root = $CatalogRoot
        merged_root = $MergedRoot
        screen_root = $ScreenRoot
        selection_root = $SelectionRoot
    }
    foreach ($key in $Extra.Keys) {
        $payload[$key] = $Extra[$key]
    }
    $payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $QueueRoot "status.json") -Encoding utf8
}

function Add-QueueLog {
    param([Parameter(Mandatory = $true)][string]$Message)

    Add-Content -LiteralPath (Join-Path $QueueRoot "queue.log") -Encoding utf8 -Value ("{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message)
}

function Get-FreezeFiles {
    $fixed = @(
        $Runner,
        $Composer,
        $Merger,
        $Screen,
        $Optimizer,
        $FactorConfig,
        $PathsConfig,
        $FactorRuntime,
        $ResearchWorkflow,
        $ResearchBootstrap,
        $FactorBase,
        $V7Catalog
    )
    $researchModules = Get-ChildItem -LiteralPath $ResearchDefsRoot -Filter "research_factor_mining_*.py" -File |
        Sort-Object FullName |
        ForEach-Object { $_.FullName }
    return @($fixed + $researchModules | Select-Object -Unique)
}

function Get-SourceFingerprints {
    $hashes = [ordered]@{}
    foreach ($path in Get-FreezeFiles) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "freeze input is missing: $path"
        }
        $hashes[$path] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    return $hashes
}

function Assert-SourceFrozen {
    param([Parameter(Mandatory = $true)]$Expected)

    foreach ($path in $Expected.Keys) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "frozen source disappeared before queued launch: $path"
        }
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $Expected[$path]) {
            throw "source drift blocked queued launch: $path"
        }
    }
}

function Get-ExpansionManifest {
    param([Parameter(Mandatory = $true)][string]$Root)

    $manifests = @(Get-ChildItem -LiteralPath $Root -Recurse -Filter "factor_mining_run_manifest.json" -File -ErrorAction SilentlyContinue)
    if ($manifests.Count -ne 1) {
        throw "expected exactly one factor_mining_run_manifest.json under $Root; found $($manifests.Count)"
    }
    return Get-Content -LiteralPath $manifests[0].FullName -Raw -Encoding utf8 | ConvertFrom-Json
}

function Assert-CompleteExpansion {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][int]$ExpectedSignals,
        [Parameter(Mandatory = $true)][int]$ExpectedFamilies,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $manifest = Get-ExpansionManifest -Root $Root
    if (-not [bool]$manifest.research_only) {
        throw "$Label manifest is not research-only"
    }
    if ([int]$manifest.catalogue.signal_count -ne $ExpectedSignals) {
        throw "$Label signal count mismatch: expected $ExpectedSignals, got $($manifest.catalogue.signal_count)"
    }
    if ([int]$manifest.catalogue.family_count -ne $ExpectedFamilies) {
        throw "$Label family count mismatch: expected $ExpectedFamilies, got $($manifest.catalogue.family_count)"
    }
    if ([string]$manifest.requested_date_range.start -ne "2025-01-01" -or [string]$manifest.requested_date_range.end -ne "2026-07-30") {
        throw "$Label date range is not the frozen 2025-01-01..2026-07-30 contract"
    }
    $factorRoot = Join-Path $Root "factor_data\factors\T1430"
    $days = @(Get-ChildItem -LiteralPath $factorRoot -Recurse -Filter "*.parquet" -File -ErrorAction Stop)
    if ($days.Count -ne 381) {
        throw "$Label needs exactly 381 T1430 parquet days; found $($days.Count)"
    }
    return $manifest
}

function Test-ActiveV5Process {
    $item = Get-CimInstance Win32_Process -Filter "ProcessId = $ActiveV5Pid" -ErrorAction SilentlyContinue
    if ($null -eq $item) {
        return $false
    }
    return [string]$item.CommandLine -match "factor_mining_20260803_aggregate_catalog_v5_full"
}

function Wait-ForV5Completion {
    while ($true) {
        try {
            $manifest = Assert-CompleteExpansion -Root $ActiveV5Root -ExpectedSignals 214 -ExpectedFamilies 75 -Label "aggregate-v5"
            Add-QueueLog "aggregate-v5 immutable-root audit passed"
            return $manifest
        }
        catch {
            if (-not (Test-ActiveV5Process)) {
                throw "aggregate-v5 is no longer running and did not pass its immutable-root audit: $($_.Exception.Message)"
            }
            Write-Status -Stage "waiting_for_aggregate_v5" -Extra @{ last_observation = $_.Exception.Message }
            Add-QueueLog "waiting for aggregate-v5: $($_.Exception.Message)"
            Start-Sleep -Seconds $PollSeconds
        }
    }
}

function Get-FreeMemoryGb {
    $os = Get-CimInstance Win32_OperatingSystem
    return [math]::Round($os.FreePhysicalMemory / 1MB, 2)
}

function Wait-ForMemory {
    $consecutive = 0
    while ($consecutive -lt 2) {
        $freeGb = Get-FreeMemoryGb
        if ($freeGb -ge $MinFreeGb) {
            $consecutive += 1
            Add-QueueLog "free memory ${freeGb}GB ($consecutive/2 healthy samples)"
        }
        else {
            $consecutive = 0
            Add-QueueLog "free memory ${freeGb}GB below ${MinFreeGb}GB; waiting"
        }
        if ($consecutive -lt 2) {
            Start-Sleep -Seconds $PollSeconds
        }
    }
}

function Invoke-PythonStage {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $stdout = Join-Path $QueueRoot ("{0}.stdout.log" -f $Stage)
    $stderr = Join-Path $QueueRoot ("{0}.stderr.log" -f $Stage)
    Add-QueueLog "starting $Stage"
    & $Py @Arguments 1> $stdout 2> $stderr
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage failed with exit code $LASTEXITCODE; see $stdout and $stderr"
    }
    Add-QueueLog "completed $Stage"
}

function Start-Queue {
    Assert-ScratchChild -Path $QueueRoot
    if (-not (Test-Path -LiteralPath $QueueRoot)) {
        New-Item -ItemType Directory -Path $QueueRoot -Force | Out-Null
    }
    $lock = Join-Path $QueueRoot "queue.lock"
    if (Test-Path -LiteralPath $lock) {
        throw "queue lock already exists: $lock"
    }
    New-Item -ItemType File -Path $lock -Force | Out-Null

    Assert-Absent -Path $V8Root
    Assert-Absent -Path $CatalogRoot
    Assert-Absent -Path $MergedRoot
    Assert-Absent -Path $ScreenRoot
    Assert-Absent -Path $SelectionRoot

    $fingerprints = Get-SourceFingerprints
    [ordered]@{
        schema_version = 1
        research_only = $true
        created_at = (Get-Date).ToString("o")
        source_fingerprints = $fingerprints
        required_inputs = [ordered]@{
            active_v5_root = $ActiveV5Root
            v7_merged_root = $V7MergedRoot
            v7_catalog = $V7Catalog
        }
    } | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $QueueRoot "queue_plan.json") -Encoding utf8

    Write-Status -Stage "waiting_for_aggregate_v5"
    Wait-ForV5Completion | Out-Null
    Assert-SourceFrozen -Expected $fingerprints
    Wait-ForMemory

    Write-Status -Stage "v8_preflight"
    Invoke-PythonStage -Stage "v8_preflight" -Arguments @(
        "-3.11", "-B", $Runner,
        "--catalog-module", "cbond_on.domain.factors.defs.research_factor_mining_daily_orthogonal_batch_v8",
        "--scratch-root", $V8Root,
        "--start", "2025-01-01", "--end", "2026-07-30"
    )

    Write-Status -Stage "v8_full_build"
    Invoke-PythonStage -Stage "v8_full_build" -Arguments @(
        "-3.11", "-B", $Runner,
        "--catalog-module", "cbond_on.domain.factors.defs.research_factor_mining_daily_orthogonal_batch_v8",
        "--scratch-root", $V8Root,
        "--start", "2025-01-01", "--end", "2026-07-30", "--execute"
    )
    Assert-CompleteExpansion -Root $V8Root -ExpectedSignals 24 -ExpectedFamilies 13 -Label "daily-orthogonal-v8" | Out-Null
    Assert-SourceFrozen -Expected $fingerprints

    Write-Status -Stage "compose_global_catalogue"
    Invoke-PythonStage -Stage "compose_global_catalogue" -Arguments @(
        "-3.11", "-B", $Composer,
        "--vetted-v3-catalog", $V7Catalog,
        "--module", "cbond_on.domain.factors.defs.research_factor_mining_aggregate_catalog_v5",
        "--module", "cbond_on.domain.factors.defs.research_factor_mining_daily_orthogonal_batch_v8",
        "--output-name", (Split-Path -Leaf $CatalogRoot), "--execute"
    )

    Write-Status -Stage "outer_union_merge"
    Invoke-PythonStage -Stage "outer_union_merge" -Arguments @(
        "-3.11", "-B", $Merger,
        "--input-root", $V7MergedRoot,
        "--input-root", (Join-Path $ActiveV5Root "factor_data"),
        "--input-root", (Join-Path $V8Root "factor_data"),
        "--output-root", $MergedRoot,
        "--family-catalog", (Join-Path $CatalogRoot "family_catalog.json"),
        "--execute"
    )

    Write-Status -Stage "fixed_global_screen"
    Invoke-PythonStage -Stage "fixed_global_screen" -Arguments @(
        "-3.11", "-B", $Screen,
        "--factor-root", $MergedRoot,
        "--label-root", "D:/cbond_on/label_data",
        "--raw-data-root", "D:/cbond_data_hub/raw_data",
        "--factor-catalog", (Join-Path $CatalogRoot "family_catalog.json"),
        "--output-dir", $ScreenRoot,
        "--panel-name", "T1430",
        "--start", "2025-01-01", "--end", "2026-07-30",
        "--top-k", "20", "--min-cross-section", "30",
        "--min-valid-days", "250", "--min-valid-days-per-partition", "50",
        "--min-redundancy-days", "200", "--ic-threshold", "0.02",
        "--within-family-threshold", "0.80", "--cross-family-threshold", "0.70",
        "--max-selected", "100"
    )

    Write-Status -Stage "exact_mis_selection"
    Invoke-PythonStage -Stage "exact_mis_selection" -Arguments @(
        "-3.11", "-B", $Optimizer,
        "--screen-dir", $ScreenRoot,
        "--output-dir", $SelectionRoot
    )

    $summaryPath = Join-Path $SelectionRoot "selection_summary.json"
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        throw "exact MIS completed without selection_summary.json: $summaryPath"
    }
    $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding utf8 | ConvertFrom-Json
    $maximum = [int]$summary.maximum_selected_count
    Write-Status -Stage "complete" -Extra @{
        maximum_selected_count = $maximum
        target_reached = ($maximum -ge 50)
        selection_summary = $summaryPath
    }
    Add-QueueLog "queue complete: exact-MIS selected $maximum factors; target 50 reached=$($maximum -ge 50)"
}

Assert-ScratchChild -Path $QueueRoot
if ($PollSeconds -lt 30) {
    throw "PollSeconds must be at least 30"
}
if ($MinFreeGb -lt 2.0) {
    throw "MinFreeGb must be at least 2.0"
}

if ($PlanOnly) {
    $plan = [ordered]@{
        research_only = $true
        active_v5_root = $ActiveV5Root
        next_catalogue = "daily_orthogonal_batch_v8 (24 signals, 13 families)"
        next_global_catalogue = "v7 + aggregate-v5 + v8 (773 signals, 165 families after preflight)"
        output_roots = @($V8Root, $CatalogRoot, $MergedRoot, $ScreenRoot, $SelectionRoot)
        required_free_memory_gb = $MinFreeGb
        source_fingerprint_file_count = (Get-FreezeFiles).Count
    }
    $plan | ConvertTo-Json -Depth 6
    exit 0
}

try {
    Start-Queue
}
catch {
    $message = "{0}: {1}" -f $_.Exception.GetType().Name, $_.Exception.Message
    if (Test-Path -LiteralPath $QueueRoot) {
        try {
            Write-Status -Stage "failed" -Extra @{ error = $message }
            Add-QueueLog "FAILED $message"
        }
        catch {
            # Preserve the original failure if queue reporting itself is unavailable.
        }
    }
    Write-Error $message
    exit 1
}
