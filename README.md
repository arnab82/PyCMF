PyCMF
==============================
[//]: # (Badges)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pycmf.readthedocs.io/en/latest/)
[![GitHub Actions Build Status](https://github.com/nmayhall-vt/pycmf/workflows/CI/badge.svg)](https://github.com/nmayhall-vt/pycmf/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/nmayhall-vt/PyCMF/branch/main/graph/badge.svg)](https://codecov.io/gh/nmayhall-vt/PyCMF/branch/main)


A CMF implementation using PySCF

Cluster mean-field (cMF) theory describes a molecule with a product of cluster states,
|Ψ⟩ = |I⟩ ⊗ |J⟩ ⊗ ..., and optimizes both the cluster states and the orbitals
([Jiménez-Hoyos & Scuseria](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.92.085101)).
This package does cMF directly on top of a PySCF mean-field object, in the AO basis, and follows
restricted open-shell cMF (RO-cMF, [Bachhar & Mayhall, J. Phys. Chem. A 2024](https://pubs.acs.org/doi/10.1021/acs.jpca.4c03914))
for open-shell clusters. It reproduces the energies of [ClusterMeanField.jl](https://github.com/nmayhall-vt/ClusterMeanField.jl).

### Quick start

```python
from pyscf import gto, scf
import pycmf

mol = gto.M(atom=[['H', (i, 0, 0)] for i in range(6)], basis='sto-3g')
mf = scf.RHF(mol).run()

cmf = pycmf.CMF(mf)
cmf.set_Cact(cmf.lowdin(), [[0, 1], [2, 3], [4, 5]], [(1, 1), (1, 1), (1, 1)])  # orbitals, clusters, (na, nb) per cluster
cmf.cmf_ci()      # cluster states at fixed orbitals:  -2.972938136549
cmf.kernel()      # orbital optimized cMF:             -3.205983033016
cmf.analyze()
```

Orbitals are given as AO coefficients: `set_Cfrz` (frozen core, not optimized), `set_Cdoc` (doubly occupied),
`set_Cact` (clusters) and `set_Cvir` (virtual).

### cMF in the AO basis

Everything that couples different clusters, and the doubly occupied core, is a mean-field interaction.
With `D_I = C_I γ_I C_I^T` the AO density of cluster `I` and `D` the total density,

    E = E_nuc + tr(D h) + 1/2 tr(D G[D]) + Σ_I [ 1/2 Σ (tu|vw)_I Γ^I_tuvw - 1/2 tr(D_I G[D_I]) ],    G[D] = J[D] - K[D]/2

and each cluster is solved with FCI for `h_I = C_I^T (h + G[D - D_I]) C_I` plus its own `(tu|vw)_I`. The orbital
gradient is `g_pq = 2 (F_pq - F_qp)` with the generalized Fock matrix built from the same AO quantities.
So the only steps are

* J/K builds with `mf.get_jk`, which means direct, in-core, density-fitted (`mf.density_fit()`) and
  scalar-relativistic (`scf.sfx2c`) integrals all work, and
* small transformations `(tu|vw)` and `(μu|vw)` for the orbitals of one cluster at a time.

The integrals of the whole active space are never built, so the number of clusters is not limited by `N_act^4` storage.
The orbitals are optimized with L-BFGS, using the exact gradient of `C = C0 exp(κ)`, and every energy evaluation
first converges the cluster states (CMF-CI).

**Open-shell clusters (RO-cMF).** Each cluster is the ensemble over the `M_s` components of its spin multiplet,
so the energy only needs spin-summed 1RDMs and spin-free 2RDMs, the embedding `J - K/2` is spin independent,
the cluster states are spin pure, and the orbitals are restricted. `fspace` only selects the `M_s` sector a cluster is solved in.

### Clusters from atomic orbitals (mixed valence, antiferromagnetic coupling)

For multi-center systems the mean-field orbitals are often delocalized (mixed valence) or spin polarized
(broken-symmetry UHF), which makes localizing them onto the centers difficult. Instead the clusters can be
taken straight from orthogonalized AOs, with the charge and spin of each center set by `fspace`:

```python
cmf = pycmf.CMF(mf)                       # mf: any converged RHF/ROHF/UHF, only its density is used
cmf.set_clusters_from_ao([['0 Fe 3d'], ['1 Fe 3d']], [(5, 1), (5, 0)])
cmf.kernel()
```

The other orbitals are split into doubly occupied and virtual ones from the density of `mf`, and the orbital
optimization relaxes everything. See `examples/mixed_valence_h4.py` (a toy mixed-valence system where ROHF delocalizes
the charge and cMF localizes it) and `examples/fe2_mixed_valence.py` (an Fe₂(μ-H)₂ complex). The coupling between
the charge-localized (or spin-coupled) cluster states is intercluster correlation, see `qdpt2` below or the export
to FermiCG.jl.

### State averaging (mixed valence)

Orbitals optimized for one charge-localized configuration are biased against its mirror image. In
`examples/mixed_valence_h4.py` at 3 Å, the orbitals of H₂ + H₂⁺ put H₂⁺ + H₂ 4.7 mEh too high.
`state_average_` optimizes one set of orbitals for the weighted average energy of several product states,
each with its own cluster states:

```python
cmf.set_clusters_from_ao([['0 Fe 3d'], ['1 Fe 3d']], [(5, 1), (5, 0)])
cmf.state_average_([[(5, 1), (5, 0)], [(5, 0), (5, 1)]])   # weights=[...] optional, equal by default
cmf.kernel()              # e_tot: average energy, e_states: energy of each state
cmf.select_state(1)       # RDMs, analyze() and export for state 1
```

In the toy example the two states come out exactly degenerate, each 1.0 mEh above the state-specific optimum.
Each extra state costs one CMF-CI and one J/K build per orbital step.

### Post-cMF in FermiCG.jl

`export_fermicg` writes the active-space Hamiltonian in the optimized orbitals (dressed by the doubly occupied
orbitals), the clusters, their Fock sectors and the embedding 1RDM to an npz file, so the inter-cluster
correlation (TPSCI, BST, PT2) can be added with [FermiCG.jl](https://github.com/mayhallgroup/FermiCG):

```python
cmf.export_fermicg('cmf.npz')
```

```julia
using NPZ, QCBase, RDM, InCoreIntegrals, FermiCG
d = npzread("cmf.npz")
ints = InCoreInts(first(d["h0"]), d["h1"], d["h2"])
clusters = [MOCluster(i, findall(==(i), d["cluster_index"])) for i in 1:maximum(d["cluster_index"])]
init_fspace = FockConfig([Tuple(d["fspace"][i, :]) for i in 1:size(d["fspace"], 1)])
d1 = RDM1(d["rdm1a"], d["rdm1b"])

cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [1, 1], init_fspace, max_roots=20)
clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
cluster_ops   = FermiCG.compute_cluster_ops(cluster_bases, ints)
FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

ci_vector = FermiCG.TPSCIstate(clusters, init_fspace, R=4)
# for a state-averaged run, add the other configurations, e.g. the mirror one
# FermiCG.add_fockconfig!(ci_vector, FockConfig([Tuple(d["fspace_states"][2, i, :]) for i in 1:length(clusters)]))
ci_vector = FermiCG.add_spin_focksectors(ci_vector)
e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, thresh_cipsi=1e-3, thresh_foi=1e-5, thresh_asci=-1)
```

For large systems, export only the clusters that are strongly coupled, e.g. the two metals:
`cmf.export_fermicg('pair.npz', clusters=[0, 1])` folds all other clusters, in their cMF states, into `h0` and `h1`
(`get_active_hamiltonian` returns the same Hamiltonian in Python). The cMF state of the pair has the total cMF energy
in that Hamiltonian, so anything solved on top of it gives the correlation within the pair in the field of the rest.

`[1, 1]` lets each cluster gain or lose one electron, which the hopping between the metals needs. The other keys
are `fspace_states`, `weights`, `e_cmf`, `e_states`, `mo_coeff` (AO coefficients of [frozen, doubly occupied,
active, virtual]) and `ncore`.

### Perturbation theory between clusters (QDPT2)

`cmf.qdpt2(clusters)` adds the correlation between a few clusters (default: all) in the mean field of the rest.
Product states degenerate with the cMF reference in zeroth order form the model space: all Ms components of the
cluster spin multiplets and, for `state_average_`, the other charge configurations. So spin coupling and electron
hopping are treated exactly, and everything else (charge transfer, dispersion) to second order.
The default Epstein-Nesbet denominators include the electron-hole attraction of charge-transfer states, which
kinetic exchange needs.

```python
res = cmf.qdpt2([0, 1])             # e.g. the two metals
res.energies, res.spins             # spin ladder (and mixed-valence states)
res.heisenberg_j()                  # J of H = -2J S_A.S_B
e, increments = cmf.pt2_mbe(order=3, cutoff=6.0)    # pairs + triples for large systems
```

| test (FCI over the same active orbitals) | QDPT2 | FCI |
| --- | --- | --- |
| N···N 2.5 Å, two S = 3/2 clusters: J | -186 cm⁻¹ | -190 cm⁻¹ |
| N···N 2.0 Å: J | -1046 cm⁻¹ | -1149 cm⁻¹ |
| (H₂)₂⁺ 3 Å, hopping splitting | 16.03 mEh | 16.03 mEh |
| H₆, three clusters: correlation recovered | 64 % (pairs only: 58 %) | 100 % |

The couplings are projected on the eigenstates of each cluster, so clusters need to be small enough to diagonalize
(about 8 orbitals); larger clusters use an iterative solver with the embedded-cluster H0 (`partition='cmf'`).

For many clusters, `cmf.pt2_mbe(order=3, cutoff=...)` expands the QDPT2 energy over pairs and triples of clusters
(each in the mean field of the rest): E = E_cMF + Σ ΔE_IJ + Σ (ΔE_IJK − ΔE_IJ − ΔE_IK − ΔE_JK). The triples add
excitations of three clusters at once, e.g. charge transfer between two clusters with the polarization of a third,
and the non-additivity of the pairs, which matters for crowded clusters. A triple of 4-orbital clusters takes about
2 s. Octatetraene (STO-3G) in 4 clusters, error with respect to QDPT2 of all clusters: pairs 0.64 mEh, pairs +
triples 0.023 mEh (with three clusters the expansion is exact). `pair_pt2` is the `order=2` truncation.

### Dynamic correlation of the environment (cMF-NEVPT2)

`cmf.nevpt2()` adds strongly contracted NEVPT2 for the excitations involving the doubly occupied and virtual
orbitals. The zeroth-order Hamiltonian is Dyall's, with its active part replaced by the sum of the embedded cluster
Hamiltonians, which has the cMF state as an eigenfunction (as NEVPT2 requires) and is the same H0 as in `qdpt2`.
So the two corrections do not overlap:

    E = E_cMF + (QDPT2 energy - E_cMF)  +  E_NEVPT2
        clusters  correlation between clusters  environment

For open-shell clusters the reference is the RO-cMF Ms ensemble (`spin_average=True`, default). The energy
expressions are those of `pyscf.mrpt.nevpt2`: with one cluster, cMF-NEVPT2 is SC-NEVPT2 of CASCI (tested), and two
separated molecules give twice the energy of one (tested). Frozen orbitals (`set_Cfrz`) are not correlated. With
density fitting all integrals come from two 3-index tensors, (L|pt) for active t and (ia|L) for core i and virtual a,
`8 N_aux (n_MO n_act + n_core n_virt)` bytes (1 GB for C₄₀H₄₂/6-31G), built once and shared by all sub-problems of
`nevpt2_mbe`; (ia|jb) and (pt|ia) are only formed in blocks (stored, (ia|jb) would be (n_core n_virt)², about 290 GB
for the Fe₂ complex). The active-space RDMs of the cMF state are built in the determinant space of the active
orbitals, so `nevpt2` needs a moderate active space (about 10-12 orbitals).

For larger active spaces, `cmf.nevpt2_mbe(order=2, cutoff=...)` expands cMF-NEVPT2 over the clusters: the
core-core/virtual-virtual class once, plus the NEVPT2 increment of each cluster and each pair of clusters, with the
other clusters as frozen spectators. Each term only needs the active space of one or two clusters, and with
orbital-optimized cMF the pair truncation is within 0.1 mEh of the full cMF-NEVPT2:

| π system (6-31G) | cMF-NEVPT2 | clusters only | clusters + pairs |
| --- | --- | --- | --- |
| hexatriene, 3 clusters | -0.460688 | 14.3 mEh error | 0.04 mEh error |
| octatetraene, 4 clusters | -0.612716 | 22.1 mEh error | 0.08 mEh error |

`order=3` adds the three-cluster terms (exact for three clusters). For the π systems above they are small, so the
pairs usually suffice; a triple of 4-orbital clusters is a 12-orbital NEVPT2, about 7 minutes on 4 cores, nearly all
in the 3- and 4-RDMs of the reference.

### RDMs and properties

| method | returns |
| --- | --- |
| `make_rdm1(ao_repr=True)` | spin-summed 1RDM of the whole system in the AO basis (dipoles, populations, natural orbitals, ...) |
| `make_rdm1s(ao_repr=True)` | α and β 1RDMs of the product state with each cluster in its `fspace` M_s sector (spin densities); `spin_average=True` gives the RO-cMF ensemble |
| `make_rdm1()`, `make_rdm12()` | active-space spin-summed 1RDM and spin-free 2RDM (PySCF convention), consistent with `e_tot` |
| `make_rdm12s()` | active-space spin-resolved 1RDMs and 2RDMs (aa, ab, bb) of the product state |
| `spin_square()` | ⟨S²⟩ of the product state and ⟨S_I²⟩ of each cluster |
| `analyze()` | cluster occupations, ⟨S_I²⟩, meta-Löwdin charges and spin populations, dipole moment |

For example `mf.dip_moment(mol, cmf.make_rdm1(ao_repr=True))` or `pyscf.tools.molden.from_mo(mol, 'cmf.molden', cmf.mo_coeff)`.

These are mean-field between the clusters. With the second-order coupling between clusters, use the first-order
wavefunction of QDPT2, Ψ = Ψ₀ + Ψ₁ for a state of the model space (`with_rdm=True`): charge transfer between the
clusters enters at first order (density between clusters, bond orders), and the spin coupling exactly (a singlet of
two S = 3/2 clusters has no spin density, unlike the broken-symmetry product state). These are RDMs of a
wavefunction, so they are N-representable, and spin eigenstates stay spin pure.

```python
res = cmf.qdpt2([0, 1], with_rdm=True)
dm_a, dm_b = res.make_rdm1s(state=0, ao_repr=True)      # whole system, AO basis
dm1, dm2 = res.make_rdm12(state=0)                      # orbitals res.orbs of the selected clusters
res.analyze(state=0)                                    # charges, spin populations, dipole
e, inc, dm1, dm1_ao = cmf.pt2_mbe(order=3, cutoff=6.0, with_rdm=True)   # spin-summed 1RDM for large systems
```

For H₄ in two clusters the largest error of the active 1RDM with respect to FCI drops from 0.11 (cMF) to 0.08.
The NEVPT2 correction is not included in these densities.

### Large systems

Per energy evaluation (one orbital step) the AO work is fixed, independent of the number of clusters
and of the number of CMF-CI sweeps:

* a J/K build for the core density and a transformation of the active-orbital integrals `(aa|aa)`,
* the CMF-CI sweeps, which only use active-orbital quantities,
* a J/K build for the converged cluster densities and one pass for the `(μu|vw)` of all clusters (gradient).

The densities are passed to `mf.get_jk` together with their orbitals, so density-fitted K costs
`O(N_aux N_AO² N_occ)`. `(aa|aa)` needs `8 n_act⁴` bytes (0.8 GB for 100 active orbitals, 13 GB for 200).

Use density fitting, `mf = scf.RHF(mol).density_fit()`. Its 3-index integrals, `8 N_aux N_AO²/2` bytes, have to fit in memory:
for a 200-atom organic molecule that is about 9 GB in STO-3G, 30 GB in 6-31G and 90-140 GB in def2-SVP
(`weigend` or `def2-universal-jkfit`). Without density fitting, one direct J/K build for the 1116-AO Fe₂ complex
takes about 12 minutes on 4 cores.

Measured on 4 cores (15 GB): all-trans polyene C₁₀₀H₁₀₂ (202 atoms, STO-3G, 602 AOs, `weigend` density fitting),
the π system split into 25 clusters of 4 orbitals, i.e. a 100-orbital active space with 301 doubly occupied and
201 virtual orbitals (115,501 orbital rotations):

| step | time |
| --- | --- |
| RHF | 530 s |
| CMF-CI at the initial orbitals (core J/K, (aa\|aa), all sweeps) | 49 s |
| one orbital step (energy + gradient) | 162 s, about 80 % in the two J/K builds |
| orbital optimization to \|g\| < 1e-5 | 30 iterations, 39 evaluations, 1.8 h |

The energy was converged to 1e-6 Eh after 11 iterations. On a node with more cores the J/K builds scale down accordingly.

With perturbation theory on top, all-trans C₄₀H₄₂ (82 atoms, 6-31G, 444 AOs, `weigend`), the π system in 10 clusters
of 4 orbitals (40 active orbitals, 121 doubly occupied, 283 virtual), pair terms for clusters within 6 Å, 4 cores:

| step | time | energy (Eh) |
| --- | --- | --- |
| RHF | 86 s | -1538.304686 |
| ooCMF to \|g\| < 1e-5 | 50 min | -1538.825585 |
| `pair_pt2(cutoff=6)`: QDPT2 between clusters, 9 pairs | 6 s | -0.040244 |
| `nevpt2_mbe(cutoff=6)`: cMF-NEVPT2, 10 clusters + 9 pairs | 6 min | -3.042435 |

The NEVPT2 time is the shared 3-index integrals (26 s), the core-core/virtual-virtual class (37 s), and 11 s per
cluster and 19 s per pair, which are dominated by matrix multiplications (`O(n_core n_virt² N_aux n_act)`); with the
cutoff the number of pairs grows linearly with the length of the system.

### Copyright

Copyright (c) 2024, Your name (or your organization/company/team)


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
