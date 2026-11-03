# TPSCI-PDFT for the Faraday Cr2 data: status

Goal: PDFT on the TPSCI states of [Cr2(OH)3(tmtacn)2]3+ from
github.com/arnab82/2024_faraday_discussions_data (`cr2/`, (32e, 38o), M = 100, `m100/tpsci_05.jl.scr/data_tpsci_04.jld2`,
97,357 configurations, 4 roots S = 0..3), and J against PT1-PDFT (-28 .. -33 cm-1, README) and experiment (-66 cm-1, H = -2J S_A.S_B).

## Scripts

- `tpsci_rdms_from_jld2.jl` (Julia, TPSChem.jl `cepa_optimization`): rebuilds the cluster bases, computes the root-diagonal
  1RDMs and spin-free 2RDMs, checks their energies against `e0a`, writes `<out>.npz`. Checkpoints `<out>_1rdm.npz`,
  `<out>_inter.npz`, `<out>_state.jld2` let a rerun skip finished steps. Options: `TPSCI_M` (100), `TPSCI_DELTA` (3),
  `TPSCI_REDIAG=1` (see below), `TPSCI_1RDM=trace|tpschem`, `TPSCI_ONCLUSTER=natural|table`.
- `tpsci_rdm_tools.jl`: the 2RDM kernels. Inter-cluster blocks: TPSChem's loop, factorized per Fock-sector pair (BLAS),
  threaded over the 620 cluster combinations (9 min on 4 threads for the Cr2 state; TPSChem's own loop ~10+ h).
  On-cluster blocks: natural cluster states, 2RDM as Gram matrices of two-hole vectors (0.4 s per state of a 10-orbital
  (5,4) cluster vs 26 s with FCI.compute_rdm1_rdm2).
- `test_tpsci_rdm_tools.jl`: checks against TPSChem on H12 (all to 1e-12; RDM energies = TPSCI energies to 1e-12).
- `cr2_tpsci_pdft.py` (PySCF/PyCMF): rebuilds the active space of the data (checked against `ints_h0/h1`), TPSCI J from
  the RDM energies, TPSCI-PDFT J for 12 functionals (`pdft.functional_scan`), writes `<rdms>_pdft.json`.

## The open problem: cluster-state signs

The rebuilt cluster bases have the same cluster-state energies as the 2024 run (all 8658 to 1.2e-6) but not the same
eigenvector signs (the 2024 run used MKL; signs of eigenvectors are arbitrary). The saved TPSCI coefficients are then not
the eigenvector in the rebuilt basis: RDM energies come out 0.070-0.076 Eh above `e0a`, root order reversed
(`tpsci_rdms_04.npz` from the cloud run is therefore NOT usable).

Fix in the script: `TPSCI_REDIAG=1` diagonalizes H again in the same 97,357 configurations (the eigenvectors are the same
states in the new sign convention; validated on a sign-scrambled H12 state, energies back to 1e-11). It uses
`TPSChem.tps_ci_davidson`, whose matrix-free `tps_ci_matvec` is serial and slow: no Davidson iteration finished in 40 min
on 4 cores in the cloud. The 2024 run stored the dense H of this space instead (`max_mem_ci = 700`, 76 GB).

Options:
1. A machine with >= 90 GB RAM: replace `tps_ci_davidson` by a stored H (`TPSChem.build_full_H_parallel` or
   `tps_ci_direct`; the latter builds the dense H).
2. Sparse H of the configuration space (`TPSChem.build_H_qq_sparse(v0a, cluster_ops, clustered_ham)`), then Davidson on it.
   Memory = nonzeros x 12 B (+ 24 B per entry while building) on top of the cluster operators (~11 GB); nonzeros not yet
   counted.
3. Try to reproduce the 2024 signs (MKL on x86): run without `TPSCI_REDIAG` and look at the energy check.
4. Rerun TPSCI with TPSChem (consistent bases and state by construction); expensive at threshold 2e-4.

## Then

    python cr2_tpsci_pdft.py DATA/cr2 tpsci_rdms_04r.npz   # computes rohf_faraday.chk (def2-SVP DF-ROHF) if missing

## Queued after TPSCI-PDFT

- Spin-resolved TPSCI 2RDMs (αα, ββ, αβ separately) and the spin-spin ZFS D tensor from the M_S = 0 components.
- SOC: spin-orbit mean field with spin-flip transition 1RDMs (`TPSChem.compute_1rdm_sf`, needs M_S = ±1 sectors), then the
  spin-flip 2RDM blocks for the exact two-electron Breit-Pauli term.
