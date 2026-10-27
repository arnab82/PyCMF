"""
Tests for quasi-degenerate PT2 on top of cMF
"""
import numpy as np
import pytest
from pyscf import gto, scf, fci

import pycmf


def test_qdpt2_explicit():
    """Second-order energy (cmf partition) against an explicit sum over the eigenstates of H0 (H4, 2 clusters)"""
    mol = gto.M(atom=[['H', (1.2*i, 0, 0)] for i in range(4)], basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    cmf = pycmf.CMF(mf)
    cmf.set_Cact(cmf.lowdin(), [[0, 1], [2, 3]], [(1, 1), (1, 1)])
    cmf.conv_tol_rdm1 = 1e-12
    cmf.kernel()
    res = cmf.qdpt2(partition='cmf')
    assert abs(res.e_cmf - cmf.e_tot) < 1e-10

    h0, h1, h2, orbs = cmf.get_active_hamiltonian()

    def matrix(h1, h2):
        addr, h = fci.direct_spin1.pspace(h1, h2, 4, (2, 2), np=36)
        H = np.zeros((36, 36))
        H[np.ix_(addr, addr)] = h
        return H
    g = cmf.cluster_1rdm
    h1_0 = np.zeros_like(h1)
    h2_0 = np.zeros_like(h2)
    for i, j in ((0, 1), (1, 0)):
        si, sj = slice(2*i, 2*i+2), slice(2*j, 2*j+2)
        h1_0[si, si] = (h1[si, si] + np.einsum('pqvw,vw->pq', h2[si, si, sj, sj], g[j])
                        - .5*np.einsum('pwvq,vw->pq', h2[si, sj, sj, si], g[j]))
        h2_0[si, si, si, si] = h2[si, si, si, si]
    H, H0 = matrix(h1, h2), matrix(h1_0, h2_0)
    psi0 = pycmf.product_ci(cmf.ci, [2, 2], [(1, 1), (1, 1)]).ravel()
    e00 = psi0 @ H0 @ psi0
    e, U = np.linalg.eigh(H0)
    w = U.T @ (H @ psi0)
    q = abs(e - e00) > 1e-6
    e2 = np.sum(w[q]**2 / (e00 - e[q]))
    assert abs(res.e2 - e2) < 1e-7

    # same with the iterative solver
    assert abs(cmf.qdpt2(partition='cmf', max_dim=1).e2 - e2) < 1e-7


def test_qdpt2_spin_ladder():
    """
    Two high-spin N atoms (S = 3/2): the model space gives the spin ladder S = 0..3, and the Epstein-Nesbet
    partition reproduces the exchange coupling of FCI over the same orbitals
    """
    mol = gto.M(atom='N 0 0 0; N 0 0 2.5', basis='6-31g', spin=6, verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-12)
    cmf = pycmf.CMF(mf)
    cmf.set_clusters_from_ao([['0 N 2p'], ['1 N 2p']], [(3, 0), (0, 3)])
    cmf.kernel()
    res = cmf.qdpt2()
    assert np.allclose(sorted(res.spins), [0, 1, 2, 3], atol=1e-6)

    h0, h1, h2, orbs = cmf.get_active_hamiltonian()
    efci = np.array([fci.addons.fix_spin_(fci.direct_spin1.FCISolver(), ss=s*(s+1)).kernel(h1, h2, 6, (3, 3), ecore=h0)[0]
                     for s in range(4)])
    order = np.argsort(res.spins)
    assert np.all(abs(res.energies[order] - efci) < 1e-3)
    assert abs(res.energies[order][3] - efci[3]) < 1e-8      # the high-spin state is exact
    j_fci = np.linalg.lstsq(np.vstack((np.ones(4), -np.arange(4.)*np.arange(1., 5.))).T, efci, rcond=None)[0][1]
    assert abs(res.heisenberg_j() - j_fci) < .05*abs(j_fci)


def test_qdpt2_mixed_valence():
    """(H2)2+ with both charge localizations in the model space: hopping splitting and energies as in FCI"""
    mol = gto.M(atom='H 0 0 0; H 0 0 .74; H 3 0 0; H 3 0 .74', basis='6-31g', charge=1, spin=1, verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-12)
    cmf = pycmf.CMF(mf)
    cmf.set_clusters_from_ao([['0 H', '1 H'], ['2 H', '3 H']], [(1, 1), (1, 0)])
    cmf.state_average_([[(1, 1), (1, 0)], [(1, 0), (1, 1)]])
    cmf.kernel()
    res = cmf.qdpt2()
    assert len(res.model) == 2
    h0, h1, h2, orbs = cmf.get_active_hamiltonian()
    efci = fci.direct_spin1.FCISolver().kernel(h1, h2, 8, (2, 1), ecore=h0, nroots=2)[0]
    assert np.all(abs(res.energies - efci) < 1e-4)


def test_pair_pt2():
    """Pair-additive PT2 for H6 with three clusters, and the distance cutoff"""
    mol = gto.M(atom=[['H', (i, 0, 0)] for i in range(6)], basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    cmf = pycmf.CMF(mf)
    cmf.set_Cact(cmf.lowdin(), [[0, 1], [2, 3], [4, 5]], [(1, 1)] * 3)
    cmf.kernel()
    e_all = cmf.qdpt2().energies[0]
    e_pairs, inc = cmf.pair_pt2()
    assert len(inc) == 3
    assert all(v < 0 for v in inc.values())
    assert cmf.e_tot > e_pairs > e_all - 5e-3
    e_near, inc_near = cmf.pair_pt2(cutoff=3.0)
    assert sorted(inc_near) == [(0, 1), (1, 2)]
    assert abs(e_near - (e_pairs - inc[(0, 2)])) < 1e-10
