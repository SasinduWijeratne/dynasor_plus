# ============================================================================
#  Dynasor+ correctness test harness (Windows PowerShell)
#
#  Runs the included FROSTT-format fixtures at several thread / rank
#  combinations and asserts that every run reports "PASS".
# ============================================================================
param([string]$Bin = "$PSScriptRoot\..\build\dynasor_plus.exe")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$tensors = @("3d_7.tns", "3d-24.tns", "4d_3_16.tns", "3d_dense.tns")
$ranks   = @(8, 16, 32, 64)
$threads = @(1, 2, 4, 8)

$fail = 0
$ran  = 0
foreach ($t in $tensors) {
    $tp = Join-Path (Join-Path $root "data") $t
    if (-not (Test-Path $tp)) { Write-Warning "skip missing $tp"; continue }
    foreach ($R in $ranks) {
        foreach ($T in $threads) {
            $ran++
            Write-Host "=== $t  rank=$R  threads=$T ==="
            $out = & $Bin $tp --rank $R --threads $T --verify 2>&1
            $out | ForEach-Object { Write-Host $_ }
            if (($out | Select-String -Quiet "^PASS") -eq $false) {
                Write-Host "*** FAILURE ***" -ForegroundColor Red
                $fail++
            }
        }
    }
}

Write-Host "============================================================"
if ($fail -eq 0) {
    Write-Host "ALL $ran CONFIGURATIONS PASSED" -ForegroundColor Green
    exit 0
} else {
    Write-Host "$fail / $ran CONFIGURATIONS FAILED" -ForegroundColor Red
    exit 1
}
