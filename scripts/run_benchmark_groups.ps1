$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$separatorIndex = [Array]::IndexOf($args, "--")

if ($separatorIndex -ge 0) {
    if ($separatorIndex -gt 0) {
        $Groups = @($args[0..($separatorIndex - 1)])
    }
    else {
        $Groups = @()
    }

    if ($separatorIndex + 1 -lt $args.Count) {
        $BenchmarkArgs = @($args[($separatorIndex + 1)..($args.Count - 1)])
    }
    else {
        $BenchmarkArgs = @()
    }
}
else {
    $Groups = @($args)
    $BenchmarkArgs = @()
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
        & python -m imcts.benchmarks --group $group --split-by-case @BenchmarkArgs
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
}
finally {
    Pop-Location
}
