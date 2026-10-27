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
the charge-localized (or spin-coupled) cluster states is intercluster correlation, left to post-cMF methods.

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

### Large systems

* Use density fitting, `mf = scf.RHF(mol).density_fit()`: both the J/K builds and the cluster integrals then use it.
* `cmf.sequential = False` builds the J/K of all clusters in one pass per CMF-CI sweep, instead of one pass per cluster.
* Per energy evaluation the cost is a few J/K builds (about one SCF iteration each) plus one integral transformation
  per cluster. As a reference, one direct (no density fitting) J/K build for the 1116-AO Fe₂ complex takes about 12 minutes on 4 cores.

### Copyright

Copyright (c) 2024, Your name (or your organization/company/team)


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
