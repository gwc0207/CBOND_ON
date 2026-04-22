param(
  [string]$Branch = "main",
  [int]$Retry = 3,
  [int]$SleepSec = 2
)

$ErrorActionPreference = "Stop"

Write-Host "[push_both] push origin/$Branch"
git push origin $Branch

for ($i = 1; $i -le $Retry; $i++) {
  Write-Host "[push_both] push github/$Branch attempt=$i/$Retry"
  git push github $Branch
  if ($LASTEXITCODE -eq 0) {
    Write-Host "[push_both] github push ok"
    exit 0
  }
  if ($i -lt $Retry) {
    Start-Sleep -Seconds $SleepSec
  }
}

Write-Host "[push_both] github push failed after retries" -ForegroundColor Red
exit 1
