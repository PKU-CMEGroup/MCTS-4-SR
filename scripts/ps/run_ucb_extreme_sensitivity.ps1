$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$BenchmarkArgs = @($args)

Push-Location $repoRoot
try {
    $Configs = Get-ChildItem "imcts/benchmarks/experiments/ucb_extreme/model_*.yaml" | Sort-Object Name

    foreach ($config in $Configs) {
        $model = $config.BaseName.Split("_")[-1].ToUpper()

        Write-Host ""
        Write-Host "=== Running UCB-extreme Model $model ==="

        & python -m imcts.benchmarks --group Nguyen --config $config.FullName @BenchmarkArgs
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }

        Write-Host ""
        Write-Host "--- Summary for Model $model ---"
        & python -m imcts.benchmarks.report Nguyen --config $config.FullName --level group
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Summary skipped: no reportable CSV output found yet."
        }
    }
}
finally {
    Pop-Location
}
