## ============================================================================
##  CP-ALS benchmark runner: Dynasor+ Gauss-Seidel vs Jacobi vs ALTO
##
##  For each tensor and each rank, runs all three drivers for a FIXED number
##  of iterations (no early-stop) so the per-iter MTTKRP cost is comparable.
##  Also records the final fit so we can sanity-check that each driver is
##  actually fitting the tensor.
##
##  Invoke with:
##      powershell -File tools\bench_cpals.ps1
## ============================================================================
$ErrorActionPreference = "Continue"
$env:DYN_CP_KERNEL = $null
$env:DYN_ALL_MODES = $null

$ITERS   = 10
$THREADS = 16

function Run-Dyn($tensor, $rank, $kernel) {
    $o = .\build\dynasor_plus.exe $tensor --decompose cpals `
        --rank $rank --threads $THREADS `
        --cp-iters $ITERS --cp-tol 0 `
        --cp-kernel $kernel 2>&1
    $line = $o | Select-String "^\[cpals\] total" | Select-Object -First 1
    $fit  = ($o | Select-String "^\[cpals\] done" | Select-Object -First 1)
    if ($line -and $fit) {
        if ($line.ToString() -match "total = ([\d.]+) s\s+spMTTKRP = ([\d.]+) s") {
            $total = [double]$Matches[1]
            $mttk  = [double]$Matches[2]
        } else { $total = 0; $mttk = 0 }
        $finalFit = 0
        if ($fit.ToString() -match "final_fit=([\d.]+)") {
            $finalFit = [double]$Matches[1]
        }
        [pscustomobject]@{ total_s = $total; mttkrp_s = $mttk; final_fit = $finalFit }
    } else { [pscustomobject]@{ total_s = 0; mttkrp_s = 0; final_fit = 0 } }
}

function Run-Alto($tensor, $rank) {
    $o = .\competitor\ALTO\cpd64.exe -i $tensor -r $rank -m $ITERS -e 1e-20 -l ALS 2>&1
    $total = 0; $mttk = 0; $finalFit = 0
    $cpd = $o | Select-String "CPD \(ALTO\):" | Select-Object -First 1
    if ($cpd -and $cpd.ToString() -match "CPD \(ALTO\):\s+([\d.]+) s") {
        $total = [double]$Matches[1]
    }
    $total_mttk = $o | Select-String "Total time \(for MTTKRP\)" | Select-Object -First 1
    if ($total_mttk -and $total_mttk.ToString() -match "\(for MTTKRP\):\s+([\d.]+)") {
        $mttk = [double]$Matches[1]
    }
    $lastFit = $o | Select-String "^it: .*fit:" | Select-Object -Last 1
    if ($lastFit -and $lastFit.ToString() -match "fit:\s*([\d.]+)") {
        $finalFit = [double]$Matches[1]
    }
    [pscustomobject]@{ total_s = $total; mttkrp_s = $mttk; final_fit = $finalFit }
}

$tensors = @(
    @{ path="data\synth_cp_3d_r16.tns";   N=3; ranks=@(16) },
    @{ path="data\synth_cp_4d_r8.tns";    N=4; ranks=@(8)  },
    @{ path="data\dense3d.tns";           N=3; ranks=@(16,32) },
    @{ path="data\bench_3d_10M.tns";      N=3; ranks=@(16,32,64) },
    @{ path="data\bench_4d_10M.tns";      N=4; ranks=@(16,32,64) },
    @{ path="data\bench_4d_30M.tns";      N=4; ranks=@(16,32)    }
)

Write-Host ("{0,-30} {1,5} {2,6}   {3,-26} {4,-26} {5,-26}" -f `
    "Tensor","N","Rank","Gauss-Seidel","Jacobi","ALTO")
Write-Host ("{0,-30} {1,5} {2,6}   {3,-26} {4,-26} {5,-26}" -f `
    "","","","total / mttkrp / fit","total / mttkrp / fit","total / mttkrp / fit")
Write-Host ("-" * 130)

foreach ($t in $tensors) {
    foreach ($r in $t.ranks) {
        $gs = Run-Dyn $t.path $r "gauss-seidel"
        $js = Run-Dyn $t.path $r "jacobi"
        $al = Run-Alto $t.path $r
        $gsStr = "{0,7:N3}/{1,6:N3}/{2,5:N3}" -f $gs.total_s, $gs.mttkrp_s, $gs.final_fit
        $jsStr = "{0,7:N3}/{1,6:N3}/{2,5:N3}" -f $js.total_s, $js.mttkrp_s, $js.final_fit
        $alStr = "{0,7:N3}/{1,6:N3}/{2,5:N3}" -f $al.total_s, $al.mttkrp_s, $al.final_fit
        Write-Host ("{0,-30} {1,5} {2,6}   {3,-26} {4,-26} {5,-26}" -f `
            (Split-Path $t.path -Leaf), $t.N, $r, $gsStr, $jsStr, $alStr)
    }
}
