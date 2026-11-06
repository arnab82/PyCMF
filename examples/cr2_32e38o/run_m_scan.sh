#!/bin/bash
# TPSCI of the (32e, 38o) Cr2 problem for several M (cluster states per Fock sector), one directory per M, then
# TPSCI-PDFT with 12 on-top functionals on all saved RDMs.
#
#   TPSCHEM=/path/to/TPSChem.jl ./run_m_scan.sh svp "25 50 100 150"
#
# Environment: TPSCHEM (TPSChem.jl project, required), JULIA (default julia), THREADS (default auto), and the
# TPSCI_* settings of tpsci_m.jl (TPSCI_THRESHS, TPSCI_PT2, TPSCI_RDMS, TPSCI_MAX_MEM_CI, ...).
set -euo pipefail
dir=${1:?usage: run_m_scan.sh <integral directory> "<M values>"}
Ms=${2:-"25 50 100"}
here=$(cd "$(dirname "$0")" && pwd)
: "${TPSCHEM:?set TPSCHEM to the TPSChem.jl directory}"
JULIA=${JULIA:-julia}
THREADS=${THREADS:-auto}
for M in $Ms; do
    out="$dir/M$M"
    mkdir -p "$out"
    echo "=== M = $M -> $out"
    TPSCI_M=$M "$JULIA" --project="$TPSCHEM" -t "$THREADS" "$here/tpsci_m.jl" "$dir/cmf_32e38o.npz" "$out" 2>&1 | tee -a "$out/tpsci.log"
done
python "$here/tpsci_pdft.py" "$dir" $(for M in $Ms; do ls "$dir/M$M"/rdms_t*.npz 2>/dev/null; done)
