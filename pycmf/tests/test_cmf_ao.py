"""
Tests for the AO basis CMF implementation
"""
import numpy as np
import scipy.linalg
import pytest
from pyscf import gto, scf, fci, mcscf

import pycmf


def test_h6_julia_reference():
    """Linear H6/STO-3G in Lowdin orbitals, same numbers as test/test_cmf.jl in ClusterMeanField.jl"""
    mol = gto.M(atom=[['H', (i, 0, 0)] for i in range(6)], basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    cmf = pycmf.CMF(mf)
    cmf.set_Cact(cmf.lowdin(), [[0, 1], [2, 3], [4, 5]], [(1, 1)] * 3)

    assert abs(cmf.cmf_ci() - -2.97293813654926351) < 1e-9
    assert abs(cmf.kernel() - -3.205983033016) < 1e-9
    assert cmf.converged


def test_single_cluster_is_casscf():
    """With one cluster CMF-CI is CASCI and the orbital optimized CMF is CASSCF"""
    mol = gto.M(atom='N 0 0 0; N 0 0 1.3', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    ncore, ncas = 4, 6
    C = mf.mo_coeff
    cmf = pycmf.CMF(mf)
    cmf.set_Cdoc(C[:, :ncore])
    cmf.set_Cact(C[:, ncore:ncore+ncas], [list(range(ncas))], [(3, 3)])
    cmf.set_Cvir(C[:, ncore+ncas:])

    e_casci = mcscf.CASCI(mf, ncas, 6).kernel()[0]
    assert abs(cmf.cmf_ci() - e_casci) < 1e-10

    mc = mcscf.CASSCF(mf, ncas, 6)
    mc.conv_tol = 1e-12
    e_casscf = mc.kernel()[0]
    assert abs(cmf.kernel() - e_casscf) < 1e-8


def test_rohf_limit():
    """All open shells in one high-spin cluster with ROHF orbitals gives the ROHF energy and a zero gradient"""
    mol = gto.M(atom='O 0 0 0; O 0 0 1.2', basis='6-31g', spin=2, verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-12)
    C = mf.mo_coeff
    ndoc = np.count_nonzero(mf.mo_occ == 2)
    nsoc = np.count_nonzero(mf.mo_occ == 1)
    cmf = pycmf.CMF(mf)
    cmf.set_Cdoc(C[:, :ndoc])
    cmf.set_Cact(C[:, ndoc:ndoc+nsoc], [list(range(nsoc))], [(nsoc, 0)])
    cmf.set_Cvir(C[:, ndoc+nsoc:])

    assert abs(cmf.cmf_ci() - mf.e_tot) < 1e-10
    assert np.linalg.norm(cmf.orbital_gradient()) < 1e-6


def _n2_quartets(fspace=[(3, 0), (2, 1)]):
    """Stretched N2, one high-spin cluster on each N, clusters built from atomic orbitals"""
    mol = gto.M(atom='N 0 0 0; N 0 0 2.5', basis='6-31g', spin=6, verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-12)
    cmf = pycmf.CMF(mf)
    cmf.set_clusters_from_ao([['0 N 2p'], ['1 N 2p']], fspace)
    cmf.conv_tol_rdm1 = 1e-11
    return cmf


def test_energy_matches_mo_rdms():
    """The AO energy equals the energy from the active space integrals and the assembled CMF RDMs"""
    cmf = _n2_quartets()
    e = cmf.cmf_ci()
    cmf.build_active_integrals()
    d1, d2 = cmf.make_rdm12()
    e_mo = cmf.active_h0 + np.einsum('pq,pq', cmf.active_h1, d1) + .5*np.einsum('pqrs,pqrs', cmf.active_h2, d2)
    assert abs(e - e_mo) < 1e-10


def test_rdms_and_spin():
    """
    AO densities, spin densities and <S^2> of the product state. Its energy from the spin resolved RDMs
    differs from the RO-cMF (Ms ensemble) energy by the exchange between the cluster spin densities.
    """
    for fspace, ss_ref in (([(3, 0), (3, 0)], 12.0), ([(3, 0), (0, 3)], 3.0)):
        cmf = _n2_quartets(fspace)
        e = cmf.cmf_ci()
        dma, dmb = cmf.make_rdm1s(ao_repr=True)
        assert abs(np.einsum('ij,ji', dma + dmb, cmf.S) - 14) < 1e-10
        assert abs(np.einsum('ij,ji', dma - dmb, cmf.S) - sum(na - nb for na, nb in fspace)) < 1e-10
        assert np.allclose(dma + dmb, cmf.make_rdm1(ao_repr=True))
        assert abs(cmf.spin_square()[0] - ss_ref) < 1e-8

        cmf.build_active_integrals()
        (d1a, d1b), (d2aa, d2ab, d2bb) = cmf.make_rdm12s()
        d2 = d2aa + d2bb + d2ab + d2ab.transpose(2, 3, 0, 1)
        e_ms = cmf.active_h0 + np.einsum('pq,pq', cmf.active_h1, d1a + d1b) + .5*np.einsum('pqrs,pqrs', cmf.active_h2, d2)
        m = [cmf.Cact[:, ci] @ (d1a - d1b)[np.ix_(ci, ci)] @ cmf.Cact[:, ci].T for ci in cmf.clusters]
        e_x = -.5*np.einsum('ij,ji', m[0], cmf.mf.get_k(cmf.mol, m[1]))
        assert abs(e_ms - e - e_x) < 1e-10


def test_orbital_gradient_finite_difference():
    cmf = _n2_quartets()
    cmf.cmf_ci()
    g = cmf.orbital_gradient()
    C0 = cmf.get_mo_coeffs()
    ci0 = [c.copy() for c in cmf.ci]
    d0 = [d.copy() for d in cmf.cluster_1rdm]

    pairs = [(17, 0), (11, 5), (9, 1), (11, 4), (14, 5), (16, 6)]
    h = 1e-4
    for p, q in pairs:
        e = []
        for sign in (1, -1):
            cmf.ci = [c.copy() for c in ci0]
            cmf.cluster_1rdm = [d.copy() for d in d0]
            K = np.zeros_like(g)
            K[p, q], K[q, p] = sign*h, -sign*h
            cmf.set_mo_coeffs(C0 @ scipy.linalg.expm(K))
            e.append(cmf.cmf_ci())
        assert abs((e[0] - e[1]) / (2*h) - g[p, q]) < 1e-7


def test_mixed_valence_dissociation():
    """
    (H2)2+ at 30 A: each charge-localized CMF state (H2 + H2+ or H2+ + H2) is exact,
    while ROHF delocalizes the hole over both molecules
    """
    mol = gto.M(atom='H 0 0 0; H 0 0 .74; H 30 0 0; H 30 0 .74', basis='6-31g', charge=1, spin=1, verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-12)
    e_fci = fci.FCI(mf).kernel()[0]

    energies = []
    for fspace in ([(1, 1), (1, 0)], [(1, 0), (1, 1)]):
        cmf = pycmf.CMF(mf)
        cmf.set_clusters_from_ao([['0 H', '1 H'], ['2 H', '3 H']], fspace)
        energies.append(cmf.kernel())
    assert abs(energies[0] - e_fci) < 1e-8
    assert abs(energies[1] - e_fci) < 1e-8
    assert mf.e_tot - e_fci > 1e-2


def test_density_fitting():
    """Density fitting is used for both the J/K builds and the cluster integrals"""
    mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-12)
    C = mf.mo_coeff
    cmf = pycmf.CMF(mf)
    cmf.set_Cdoc(C[:, :3])
    cmf.set_Cact(C[:, 3:7], [[0, 1], [2, 3]], [(1, 1), (1, 1)])
    cmf.set_Cvir(C[:, 7:])
    cmf.cmf_ci()

    cmf.build_active_integrals()
    d1, d2 = cmf.make_rdm12()
    eri = cmf.ao2mo((cmf.Cact,) * 4)
    e_mo = cmf.active_h0 + np.einsum('pq,pq', cmf.active_h1, d1) + .5*np.einsum('pqrs,pqrs', eri, d2)
    assert abs(cmf.energy() - e_mo) < 1e-10

    # (mu u|v w) of all clusters from a single pass over the 3-index integrals
    eye = np.eye(mol.nao)
    for ci, eri_ao in zip(cmf.clusters, cmf.get_cluster_eris_ao()):
        Ci = cmf.Cact[:, ci]
        ref = mf.with_df.ao2mo((eye, Ci, Ci, Ci), compact=False).reshape(eri_ao.shape)
        assert abs(eri_ao - ref).max() < 1e-12


def _h4_cation(R=3.0):
    mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0, 0, .74)], ['H', (R, 0, 0)], ['H', (R, 0, .74)]],
                basis='6-31g', charge=1, spin=1, verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-12)
    cmf = pycmf.CMF(mf)
    cmf.set_clusters_from_ao([['0 H', '1 H'], ['2 H', '3 H']], [(1, 1), (1, 0)])
    return cmf


def test_state_average_gradient():
    """Gradient of the state-averaged energy against finite differences"""
    cmf = _h4_cation()
    cmf.state_average_([[(1, 1), (1, 0)], [(1, 0), (1, 1)]], weights=[.7, .3])
    cmf.conv_tol_rdm1 = 1e-11
    e = cmf.cmf_ci()
    assert abs(e - (.7*cmf.e_states[0] + .3*cmf.e_states[1])) < 1e-12
    g = cmf.orbital_gradient()
    C0 = cmf.get_mo_coeffs()
    h = 1e-4
    for p, q in [(2, 0), (5, 1), (6, 3), (7, 2)]:
        es = []
        for sign in (1, -1):
            K = np.zeros_like(g)
            K[p, q], K[q, p] = sign*h, -sign*h
            cmf.set_mo_coeffs(C0 @ scipy.linalg.expm(K))
            es.append(cmf.cmf_ci())
        assert abs((es[0] - es[1]) / (2*h) - g[p, q]) < 1e-7


def test_state_average_mixed_valence():
    """
    Averaging the two charge-localized states of (H2)2+ gives orbitals shared by both: the two state energies
    are equal by symmetry and lie above the state-specific cMF energy
    """
    cmf = _h4_cation()
    e_ss = cmf.kernel()

    cmf = _h4_cation()
    cmf.state_average_([[(1, 1), (1, 0)], [(1, 0), (1, 1)]])
    e_sa = cmf.kernel()
    assert cmf.converged
    assert abs(cmf.e_states[0] - cmf.e_states[1]) < 1e-6
    assert e_ss - 1e-9 < e_sa < e_ss + 1e-2

    cmf.select_state(1)
    assert cmf.fspace == [(1, 0), (1, 1)]
    assert abs(np.trace(cmf.cluster_1rdm[0]) - 1) < 1e-10


def test_export_fermicg(tmp_path):
    """The exported active-space Hamiltonian and RDMs give back the CMF energy, with each cluster made contiguous"""
    cmf = _n2_quartets()
    perm = [0, 3, 1, 4, 2, 5]
    cmf.set_Cact(cmf.Cact[:, perm], [[0, 2, 4], [1, 3, 5]], [(3, 0), (2, 1)])
    e = cmf.cmf_ci()

    cmf.export_fermicg(str(tmp_path / 'cmf.npz'))
    d = np.load(tmp_path / 'cmf.npz')
    order = [0, 2, 4, 1, 3, 5]
    d2 = cmf.make_rdm12()[1][np.ix_(order, order, order, order)]
    e_npz = d['h0'] + np.einsum('pq,pq', d['h1'], d['rdm1a'] + d['rdm1b']) + .5*np.einsum('pqrs,pqrs', d['h2'], d2)
    assert abs(e_npz - e) < 1e-10
    assert d['cluster_index'].tolist() == [1, 1, 1, 2, 2, 2]
    assert d['fspace'].tolist() == [[3, 0], [2, 1]]
    assert np.allclose(d['mo_coeff'][:, d['ncore']:d['ncore']+6], cmf.Cact[:, order])


def test_export_subproblem(tmp_path):
    """
    A pair of clusters, with the third folded into the mean field: their cMF state has the total cMF energy
    in the exported Hamiltonian, and FCI of the pair lowers it (correlation within the pair)
    """
    mol = gto.M(atom=[['H', (i, 0, 0)] for i in range(6)], basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    cmf = pycmf.CMF(mf)
    cmf.set_Cact(cmf.lowdin(), [[0, 1], [2, 3], [4, 5]], [(1, 1)] * 3)
    e = cmf.kernel()
    d1, d2 = cmf.make_rdm12()
    for pair in ([0, 1], [1, 2], [2, 0]):
        cmf.export_fermicg(str(tmp_path / 'pair.npz'), clusters=pair)
        d = np.load(tmp_path / 'pair.npz')
        orbs = sum([cmf.clusters[i] for i in pair], [])
        e_npz = (d['h0'] + np.einsum('pq,pq', d['h1'], d['rdm1a'] + d['rdm1b'])
                 + .5*np.einsum('pqrs,pqrs', d['h2'], d2[np.ix_(orbs, orbs, orbs, orbs)]))
        assert abs(e_npz - e) < 1e-10
        assert d['nact'] == 4 and d['cluster_index'].tolist() == [1, 1, 2, 2]
        e_pair = fci.direct_spin1.FCI().kernel(d['h1'], d['h2'], 4, (2, 2), ecore=float(d['h0']))[0]
        assert e_pair < e - 1e-4


def test_ao_labels_exact_atom():
    """
    'atom-index symbol shell' labels select exactly that atom, also where PySCF's pattern search (as in older PySCF
    versions) matches '2 C 2pz' to atoms 12, 22, ...
    """
    import re
    from pycmf.cmf import _search_ao
    atoms = [['C' if i % 2 == 0 else 'H', (0.9*i, 0.3*(-1)**i, 0)] for i in range(24)]
    mol = gto.M(atom=atoms, basis='sto-3g', spin=0, verbose=0)
    labels = mol.ao_labels()
    assert [labels[i].split()[0] for i in _search_ao(mol, '2 C 2pz')] == ['2']
    assert len(_search_ao(mol, '0 C 2p')) == 3 and len(_search_ao(mol, '1 H')) == 1
    with pytest.raises(ValueError):
        _search_ao(mol, '1 C 2pz')                     # atom 1 is H
    mol.search_ao_label = lambda label: np.array([i for i, s in enumerate(labels) if re.search(label, s)])
    cmf = pycmf.CMF(scf.RHF(mol).run())
    cmf.set_clusters_from_ao([['%d C 2pz' % (2*a) for a in range(k, k + 2)] for k in range(0, 12, 2)], [(1, 1)] * 6)
    assert [len(c) for c in cmf.clusters] == [2] * 6


def test_spade_guess():
    """
    Clusters with correlating shells (2p + 3p): the SPADE guess (mean-field orbitals projected on the cluster AOs)
    starts closer to the optimized cMF than the bare AOs, and both reach the same energy
    """
    mol = gto.M(atom='N 0 0 0; N 0 0 2.5', basis='6-31g', spin=6, verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-12)
    out = {}
    for guess in ('ao', 'spade'):
        cmf = pycmf.CMF(mf)
        cmf.set_clusters_from_ao([['0 N 2p', '0 N 3p'], ['1 N 2p', '1 N 3p']], [(3, 0), (0, 3)], guess=guess)
        C = np.hstack((cmf.Cdoc, cmf.Cact, cmf.Cvir))
        assert abs(C.T @ cmf.S @ C - np.eye(C.shape[1])).max() < 1e-10
        out[guess] = (cmf.cmf_ci(), cmf.kernel())
    assert abs(out['ao'][1] - out['spade'][1]) < 1e-7
    assert out['spade'][0] - out['spade'][1] < out['ao'][0] - out['ao'][1]
