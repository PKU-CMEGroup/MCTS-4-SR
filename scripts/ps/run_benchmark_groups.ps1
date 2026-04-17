$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$Groups = @()
$BenchmarkArgs = @()
$parsingBenchmarkArgs = $false

foreach ($arg in $args) {
    if (-not $parsingBenchmarkArgs -and $arg -eq "--") {
        $parsingBenchmarkArgs = $true
        continue
    }

    if (-not $parsingBenchmarkArgs -and [string]$arg -notlike "--*") {
        $Groups += $arg
        continue
    }

    $parsingBenchmarkArgs = $true
    $BenchmarkArgs += $arg
}

Push-Location $repoRoot
try {
    if (-not $Groups -or $Groups.Count -eq 0) {
        $discoveredGroups = python -c "from imcts.benchmarks.registry import load_bundled_registry; print('\n'.join(load_bundled_registry().list_groups()))"
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        $Groups = @($discoveredGroups | Where-Object { $_.Trim() -ne "" })
    }

    foreach ($group in $Groups) {
        Write-Host ""
        Write-Host "=== Running benchmark group: $group ==="
        & python -m imcts.benchmarks --group $group @BenchmarkArgs
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
}
finally {
    Pop-Location
}
