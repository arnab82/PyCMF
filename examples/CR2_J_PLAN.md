# Cr2 exchange coupling: plan

[Cr2(OH)3(tmtacn)2]3+, J for H = -2J S_A.S_B; experiment -66 cm-1. All J values go into one table:
method x active space x M x basis x functional.

So far (def2-SVP): QDPT2 -16.45, PT1-PDFT -27.5 .. -32.7 over 12 functionals ((24e, 38o) AO clusters, ooCMF orbitals);
Faraday TPSCI(var) -18.3 .. -26.3 by threshold. TPSCI-PDFT on the Faraday state is blocked by the cluster-state sign problem
(TPSCI_PDFT_STATUS.md); running TPSCI ourselves avoids it, since the cluster bases and the state come from the same run.

## 0. Active space from cMF-SCF (running)

`cr2_active_space_selection.py`: ooCMF in (30e, 52o) (Cr 3d4d4s4p, O 2s2p3s3p), natural occupations / entropies per cluster
plus the Cr-Cr PT1 correlation, keep orbitals with min(n, 2-n) >= 0.02, ooCMF again in the selected space; QDPT2 J before
and after. Output: the selected active space and its orbitals (`cmf_reduced.npy`).

## 1. PDFT in the cMF-SCF active space

- a. QDPT2-PT1-cMF-PDFT: PT1 RDMs of each spin state in the selected space and orbitals, `pdft.functional_scan`, all
  functionals of item 4. Cheap, runs here.
- b. TPSCI-cMF-PDFT: export the selected space (`export` as in cr2_pantazis_ao.py, `cmf_tpsci.npz`), TPSCI + RDMs with
  `tpsci_rdms.jl` (4 roots, one run, so no sign problem), then `tpsci_pdft`. RDMs with the fast kernels of
  `tpsci_rdm_tools.jl` (the inter-cluster 2RDM took 9 min for 97k configurations).
- Compare with the (24e, 38o) results: does the cMF-SCF active space move J toward -66?

## 2. M dependence of TPSCI and TPSCI-PDFT

`TPSCI_M` = 25, 50, 100, 150 (and `TPSCI_THRESH_CIPSI` 1e-3 .. 2e-4 at the largest M that fits), same space and orbitals.
Report J(TPSCI var), J(TPSCI + PT2) and J(TPSCI-PDFT) per functional against M. PDFT should converge faster in M than the
wavefunction J; if it does not, the on-top densities are not converged. Memory grows with M (~12 GB at M = 100 for 97k
configurations): M >= 150 belongs on the local machine.

## 3. Basis set

- Our pipeline in def2-TZVP and cc-pVDZ (both defined for Cr in PySCF): ROHF -> ooCMF (big space) -> selection -> PT1-PDFT
  and TPSCI-PDFT. TZVP roughly doubles the AO count: ROHF and ooCMF on the local machine or a cluster node.
- DMRG-SCF-PDFT reference (-68 cm-1 reported with cc-pVDZ, to be checked against the paper: basis, active space, M,
  functional, geometry): reproduce it with PySCF + block2 (DMRG-SCF) + pyscf-forge mcpdft in cc-pVDZ, then rerun it in
  def2-SVP / def2-TZVP. This separates the basis effect from the method effect, and puts DMRG-SCF-PDFT and
  TPSCI-cMF-PDFT on the same basis and geometry.

## 4. Functionals

Every PDFT run uses the same set: tLDA, ftLDA, tPBE, ftPBE, tBLYP, ftBLYP, trevPBE, ftrevPBE, tOPBE, tM06L, tPBE0, MC23
(one grid pass per state for all), plus tPBEsol, tBP86, tPW91, tN12, tSCAN where pyscf-forge supports them. Report the
spread per method / space / basis: a method whose J depends little on the functional is the more trustworthy one.

## Order

0 (now) -> 1a (here) -> 1b and 2 (Julia; small M here, large M locally) -> 3 (locally) -> 4 runs inside every step.
