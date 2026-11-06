# Cr2 exchange coupling: plan

[Cr2(OH)3(tmtacn)2]3+, J for H = -2J S_A.S_B; experiment -66 cm-1. All J values go into one table:
method x active space x M x basis x functional.

So far (def2-SVP): QDPT2 -16.45, PT1-PDFT -27.5 .. -32.7 over 12 functionals ((24e, 38o) AO clusters, ooCMF orbitals);
Faraday TPSCI(var) -18.3 .. -26.3 by threshold. TPSCI-PDFT on the Faraday state is blocked by the cluster-state sign problem
(TPSCI_PDFT_STATUS.md); running TPSCI ourselves avoids it, since the cluster bases and the state come from the same run.

## 0. Active space from cMF-SCF (first run done)

`cr2_active_space_selection.py`, def2-SVP, AO start, Newton:

| space | E(ooCMF) / Eh | QDPT2 J (Cr-Cr pair) / cm-1 |
|---|---|---|
| (24e, 38o) Cr 3d4d, O 2p3p (cr2_pantazis_ao.py) | -3344.20813 | -16.45 |
| (30e, 52o) Cr 3d4d4s4p, O 2s2p3s3p (big) | -3344.32134 | not computed (see below) |
| (24e, 24o) selected from the big space, occ_thresh 0.02 | -3344.19091 | -13.59 (S=0..3 at 0, 27.2, 81.5, 163.1) |

Selection (cMF natural occupations): Cr (3,0) in 14 orbitals keeps only the 3 magnetic orbitals (0.998); everything else is
<= 0.0011 (entropies <= 0.009) and goes virtual. O (4,4) in 8 orbitals: 1.987 (2s-like, -> core) | 1.979 1.977 1.976 |
0.0240 0.0225 0.0205 | 0.0136 (-> virtual), entropies 0.07-0.11 throughout: no gap, the threshold cuts through the dynamic
correlation of the bridges.

Lessons:
- cMF occupations only see correlation within a cluster. A high-spin d3 ion has almost none, so the Cr 4d shell is dropped,
  although it matters for J through charge transfer / superexchange between clusters (-16.45 -> -13.59 cm-1).
- The PT1 1RDM of the Cr-Cr pair cannot be computed in the big space: QDPT2 of 14-orbital Cr clusters (largest reachable
  sector (3,2), 33124 determinants) falls back to the iterative 'cmf' partition in the 1e7-determinant pair space (> 1 h, not
  finished). In the selected space it is uninformative (3 orbitals per Cr: occupations stay 1.0).
- Needed for a selection that serves J: an inter-cluster measure in the big space, e.g. QDPT2 / PT1 in a basis of the M
  lowest cluster states per Fock sector (Davidson instead of full diagonalization of each sector), and a gap or count
  criterion for clusters with continuous occupations (the bridges).

## 1. PDFT in the cMF-SCF active space

- a. QDPT2-PT1-cMF-PDFT: PT1 RDMs of each spin state in the selected space and orbitals, `pdft.functional_scan`, all
  functionals of item 4. Cheap, runs here.
- b. TPSCI-cMF-PDFT: export the selected space (`export` as in cr2_pantazis_ao.py, `cmf_tpsci.npz`), TPSCI + RDMs with
  `tpsci_rdms.jl` (4 roots, one run, so no sign problem), then `tpsci_pdft`. RDMs with the fast kernels of
  `tpsci_rdm_tools.jl` (the inter-cluster 2RDM took 9 min for 97k configurations).
- Compare with the (24e, 38o) results: does the cMF-SCF active space move J toward -66?

## 2. M dependence of TPSCI and TPSCI-PDFT

M is the number of cluster states kept in each Fock sector of each cluster: it only enters the post-cMF TPSCI step. ooCMF,
the selection, QDPT2 and PT1-PDFT do not depend on it (QDPT2 works in the determinant space of the clusters), so the scan
reruns only cluster bases -> TPSCI -> RDMs -> TPSCI-PDFT, on one fixed active space and set of orbitals (from step 0).
`TPSCI_M` = 25, 50, 100, 150 (and `TPSCI_THRESH_CIPSI` 1e-3 .. 2e-4 at the largest M that fits). Report J(TPSCI var),
J(TPSCI + PT2) and J(TPSCI-PDFT) per functional against M. PDFT should converge faster in M than the wavefunction J; if it
does not, the on-top densities are not converged. Memory: the cluster operators grow ~M^2 per pair of Fock sectors (~12 GB
at M = 100 for the 97k-configuration Faraday state), so the small-M runs fit here and M >= 150 belongs on the local machine.

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
