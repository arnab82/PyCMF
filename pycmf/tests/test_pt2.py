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


def _h4_two_clusters():
    mol = gto.M(atom=[['H', (1.2*i, 0, 0)] for i in range(4)], basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    cmf = pycmf.CMF(mf)
    cmf.set_Cact(cmf.lowdin(), [[0, 1], [2, 3]], [(1, 1), (1, 1)])
    cmf.conv_tol_rdm1 = 1e-12
    cmf.ci_conv_tol = 1e-14
    cmf.kernel()
    return mf, cmf


def test_qdpt2_wavefunction():
    """
    First-order wavefunction (cmf partition) against an explicit sum over the eigenstates of H0 (H4, 2 clusters),
    with the projection engine and the iterative solver; its RDMs are closer to FCI than the cMF ones
    """
    mf, cmf = _h4_two_clusters()
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
    e, U = np.linalg.eigh(H0)
    w = U.T @ (H @ psi0)
    q = abs(e - psi0 @ H0 @ psi0) > 1e-6
    psi1 = U[:, q] @ (w[q] / (psi0 @ H0 @ psi0 - e[q]))

    for max_dim in (5000, 1):
        res = cmf.qdpt2(partition='cmf', max_dim=max_dim, with_rdm=True)
        phase = res.psi0[0].ravel() @ psi0
        assert abs(abs(phase) - 1) < 1e-10
        assert abs(phase*res.psi1[0].ravel() - psi1).max() < 1e-6

    res = cmf.qdpt2(with_rdm=True)
    dm_fci = fci.direct_spin1.make_rdm1(np.linalg.eigh(H)[1][:, 0].reshape(6, 6), 4, (2, 2))
    # Frobenius norm: invariant to rotations of the orbitals within a cluster, which the optimizers may apply
    assert np.linalg.norm(res.make_rdm1() - dm_fci) < np.linalg.norm(cmf.make_rdm1() - dm_fci) - 0.02
    assert abs(np.trace(res.make_rdm1()) - 4) < 1e-10
    assert abs(np.trace(res.make_rdm1(ao_repr=True) @ mf.get_ovlp()) - 4) < 1e-10
    dm1, dm2 = res.make_rdm12()
    assert abs(np.einsum('pq,pq', h1, dm1) + .5*np.einsum('pqrs,pqrs', h2, dm2) + h0
               - fci.direct_spin1.energy(h1, h2, res.wavefunction(), 4, (2, 2)) - h0) < 1e-10
    # without the first-order part, the model space is the cMF product state
    assert abs(res.make_rdm1(order=0) - cmf.make_rdm1()).max() < 1e-6


def test_qdpt2_rdm_spin_ladder():
    """
    Two high-spin N atoms: the first-order correction satisfies H_eff - H_1st = (<a|H|b1> + <b|H|a1>)/2, is
    orthogonal to the model space, and every state of the spin ladder stays a spin eigenstate
    """
    mol = gto.M(atom='N 0 0 0; N 0 0 2.5', basis='6-31g', spin=6, verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-12)
    cmf = pycmf.CMF(mf)
    cmf.set_clusters_from_ao([['0 N 2p'], ['1 N 2p']], [(3, 0), (0, 3)])
    cmf.kernel()
    res = cmf.qdpt2(with_rdm=True)
    h0, h1, h2, orbs = cmf.get_active_hamiltonian()
    h2e = fci.direct_spin1.absorb_h1e(h1, h2, res.norb, res.nelec, .5)
    HP = np.array([fci.direct_spin1.contract_2e(h2e, p, res.norb, res.nelec).ravel() for p in res.psi0])
    v = HP @ res.psi1.reshape(len(res.model), -1).T
    assert abs(res.heff - res.h1st - .5*(v + v.T)).max() < 1e-10
    assert abs(res.psi0.reshape(len(res.model), -1) @ res.psi1.reshape(len(res.model), -1).T).max() < 1e-10
    for k in range(len(res.energies)):
        s = round(res.spins[k])
        assert abs(res.spin_square(k) - s*(s+1)) < 1e-8
        dm_a, dm_b = res.make_rdm1s(k, ao_repr=True)
        assert abs(np.trace((dm_a - dm_b) @ mf.get_ovlp())) < 1e-8          # total M_s = 0
        assert abs(np.trace((dm_a + dm_b) @ mf.get_ovlp()) - mol.nelectron) < 1e-8


def test_pt2_mbe():
    """
    Many-body expansion of QDPT2 for hexatriene in three clusters: the pairs are pair_pt2, and with the triple the
    expansion reproduces QDPT2 of all clusters, energy and 1RDM
    """
    atoms = [['C', (0.000, 0.000, 0)], ['C', (1.169, 0.675, 0)], ['C', (2.425, -0.050, 0)],
             ['C', (3.594, 0.625, 0)], ['C', (4.850, -0.100, 0)], ['C', (6.019, 0.575, 0)],
             ['H', (-0.944, 0.545, 0)], ['H', (0.000, -1.090, 0)], ['H', (1.169, 1.765, 0)],
             ['H', (2.425, -1.140, 0)], ['H', (3.594, 1.715, 0)], ['H', (4.850, -1.190, 0)],
             ['H', (6.963, 0.030, 0)], ['H', (6.019, 1.665, 0)]]
    mol = gto.M(atom=atoms, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-10)
    cmf = pycmf.CMF(mf)
    cmf.set_clusters_from_ao([['0 C 2pz', '1 C 2pz'], ['2 C 2pz', '3 C 2pz'], ['4 C 2pz', '5 C 2pz']], [(1, 1)] * 3)
    cmf.kernel()
    full = cmf.qdpt2(with_rdm=True)
    dm_full = cmf.make_rdm1(ao_repr=True) + full._embed['C'] @ (full.make_rdm1() - cmf.make_rdm1()[np.ix_(full.orbs, full.orbs)]) @ full._embed['C'].T
    assert abs(full.make_rdm1(ao_repr=True) - dm_full).max() < 1e-12

    e2, inc2 = cmf.pt2_mbe(order=2)
    assert sorted(inc2) == [(0, 1), (0, 2), (1, 2)]
    assert abs(e2 - cmf.pair_pt2()[0]) < 1e-12
    assert abs(e2 - full.energies[0]) > 1e-4
    e3, inc3, dm1, dm1_ao = cmf.pt2_mbe(order=3, with_rdm=True)
    assert len(inc3) == 4
    assert abs(e3 - full.energies[0]) < 1e-10
    assert abs(dm1_ao - full.make_rdm1(ao_repr=True)).max() < 1e-10
    assert sorted(cmf.pt2_mbe(order=3, cutoff=3.0)[1]) == [(0, 1), (1, 2)]
