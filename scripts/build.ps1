# ============================================================================
#  Dynasor+  Windows PowerShell build script
#
#  Usage:
#      powershell -ExecutionPolicy Bypass -File scripts\build.ps1
#      powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -ISA avx2
#      powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Clean
#
#  Needs: g++ (MSYS2 / MinGW-w64) with OpenMP.
# ============================================================================
param(
    [string]$ISA   = "native",      # native | avx512 | avx2 | sse2 | neon | scalar
    [string]$CXX   = "g++",
    [switch]$Clean = $false
)

$ErrorActionPreference = "Stop"
$root  = Split-Path -Parent $PSScriptRoot
$src   = Join-Path $root "src"
$inc   = Join-Path $root "include"
$build = Join-Path $root "build"
$bin   = Join-Path $build "dynasor_plus.exe"

if ($Clean) {
    if (Test-Path $build) { Remove-Item -Recurse -Force $build }
    Write-Host "[build] cleaned $build"
    return
}

if (-not (Test-Path $build)) { New-Item -ItemType Directory -Path $build | Out-Null }

switch ($ISA) {
    "native"      { $arch = "-march=native -mtune=native" }
    "znver5"      { $arch = "-march=znver5 -mtune=znver5 -mprefer-vector-width=512" }
    "znver4"      { $arch = "-march=znver4 -mtune=znver4 -mprefer-vector-width=512" }
    "neoverse-v2" { $arch = "-mcpu=neoverse-v2 -msve-vector-bits=128" }
    "neoverse-v1" { $arch = "-mcpu=neoverse-v1 -msve-vector-bits=256" }
    "neoverse-n2" { $arch = "-mcpu=neoverse-n2 -msve-vector-bits=128" }
    "avx512"      { $arch = "-mavx512f -mavx512dq -mavx512vl -mfma -mprefer-vector-width=512" }
    "avx2"        { $arch = "-mavx2 -mfma" }
    "sse2"        { $arch = "-msse2" }
    "neon"        { $arch = "-march=armv8-a+simd" }
    "sve2"        { $arch = "-march=armv9-a+sve2" }
    "scalar"      { $arch = "" }
    default       { Write-Error "Unknown ISA: $ISA"; exit 1 }
}

$cxxflags = "-O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter -fopenmp -funroll-loops $arch -I`"$inc`""
$ldflags  = "-fopenmp"

$sources = @(
    "dynasor_io",
    "dynasor_preprocess",
    "dynasor_kernel",
    "dynasor",
    "dynasor_reference",
    "main"
)

$objs = @()
foreach ($s in $sources) {
    $cpp = Join-Path $src "$s.cpp"
    $obj = Join-Path $build "$s.o"
    Write-Host "[build] $CXX -c $s.cpp"
    $cmd = "$CXX $cxxflags -c `"$cpp`" -o `"$obj`""
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { Write-Error "compile failed: $s"; exit 1 }
    $objs += "`"$obj`""
}

Write-Host "[build] linking $bin"
$linkCmd = "$CXX $ldflags $($objs -join ' ') -o `"$bin`""
Invoke-Expression $linkCmd
if ($LASTEXITCODE -ne 0) { Write-Error "link failed"; exit 1 }

Write-Host "--"
Write-Host "[build] produced $bin  (ISA=$ISA)"
