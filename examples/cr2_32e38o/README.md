# Cr2 (32e, 38o): AO-cMF integrals, TPSCI M scan, TPSCI-PDFT

[Cr2(OH)3(tmtacn)2]3+, the active space of the 2024 Faraday Discussions TPSCI data: Cr 3d4d (10 orbitals, (5,2) and
(2,5)) and the 2p3p of the three bridging O (6 orbitals, (3,3) each). J is for H = -2J S_A.S_B (experiment -66 cm-1).

| step | script | what it does |
|---|---|---|
| 1 | `make_integrals.py` | ROHF, AO-cMF clusters (SPADE projection of the ROHF orbitals on the cluster AOs), ooCMF (all orbitals), active-space integrals with exact two-electron terms |
| 2 | `tpsci_m.jl` | TPSCI at one M with the Faraday threshold schedule; saves the cluster bases, the wavefunction after each threshold, and the RDMs of every root |
| 2' | `run_m_scan.sh` | step 2 for several M, then step 3 |
| 3 | `tpsci_pdft.py` | 12 on-top functionals on every RDM file, J of TPSCI, TPSCI+PT2 and each functional; `pdft_summary.csv` |

## Requirements

- Python: PySCF, pyscf-forge (`mcpdft`, for PDFT), PyCMF (this repository, `pip install -e .`)
- Julia 1.10 with TPSChem.jl (branch `cepa_optimization`), `julia --project=TPSChem.jl -e 'using Pkg; Pkg.instantiate()'`

## Commands

```bash
cd examples/cr2_32e38o
# 1. integrals (def2-SVP: 665 AOs, ~1-2 h on 4 cores; def2-TZVP: ~1200 AOs, start from the SVP ROHF, >= 32 GB)
python make_integrals.py --basis def2-svp  --out svp
python make_integrals.py --basis def2-tzvp --out tzvp --guess-chk svp/rohf.chk

# 2 + 3. TPSCI for several M, then PDFT on all RDMs
TPSCHEM=/path/to/TPSChem.jl TPSCI_MAX_MEM_CI=200 ./run_m_scan.sh svp "25 50 100 150"
TPSCHEM=/path/to/TPSChem.jl TPSCI_MAX_MEM_CI=200 ./run_m_scan.sh tzvp "50 100"

# or one M at a time
TPSCI_M=100 julia --project=/path/to/TPSChem.jl -t auto tpsci_m.jl svp/cmf_32e38o.npz svp/M100
python tpsci_pdft.py svp svp/M100/rdms_t*.npz
```

## Notes

- **Memory.** `tpsci_ci` keeps the CI Hamiltonian dense up to `TPSCI_MAX_MEM_CI` GB (default 20); beyond that it uses
  TPSChem's matrix-free Davidson, which is very slow at ~1e5 configurations. The Faraday run (M = 100, final threshold
  2e-4, 97k configurations, 76 GB dense H) set 700. Shorten the schedule (`TPSCI_THRESHS=1e-3,6e-4,4e-4`) on smaller
  machines. The cluster operators grow ~M^2 per pair of Fock sectors (~12 GB at M = 100).
- **Keep `cluster_bases.jld2` with the states.** The TPSCI coefficients refer to the cluster states of that run; cluster
  states rebuilt elsewhere (another BLAS, MKL vs OpenBLAS) can differ in sign, and the saved coefficients then no longer
  give the eigenvector (what blocked PDFT on the Faraday states, see `../TPSCI_PDFT_STATUS.md`).
- **RDMs** are root-diagonal (the roots are different spin states, between which spin-free transition RDMs vanish),
  computed with `../tpsci_rdm_tools.jl` (inter-cluster blocks threaded and factorized: 9 min for 97k configurations on
  4 threads; on-cluster blocks from natural cluster states); each file checks its RDM energies against the TPSCI ones.
- **Orbitals.** ooCMF optimizes the doubly occupied and virtual orbitals together with the active ones (the Faraday
  `cmf.jl` rotated only within the active space), so def2-SVP results differ slightly from the Faraday data. The
  two-electron integrals of the active space are exact; ROHF and ooCMF use density fitting (def2-universal-jkfit).
- **PDFT** uses density-fitted integrals for the classical and wave-function parts (`j_rdm_df` in the output is J of
  the RDM energies with those integrals, to compare with `j_tpsci_var`), grid level 3 by default.
