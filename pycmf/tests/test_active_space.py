"""
Tests for the cMF active-space selection (pycmf.active_space)
"""
import numpy as np
import pytest
from pyscf import gto, scf, fci
from pyscf.fci import cistring

import pycmf


def _be2():
    mol = gto.M(atom='Be 0 0 0; Be 0 0 2.6', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    cmf = pycmf.CMF(mf)
    cmf.set_clusters_from_ao([['0 Be 1s', '0 Be 2s', '0 Be 2p'], ['1 Be 1s', '1 Be 2s', '1 Be 2p']], [(2, 2), (2, 2)])
    cmf.algorithm = 'newton'
    cmf.conv_tol_grad = 1e-7
    cmf.kernel()
    return cmf


def _h2_dimer(r=2.0):
    mol = gto.M(atom='H 0 0 0; H 0 0 0.9; H %g 0 0; H %g 0 0.9' % (r, r), basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    cmf = pycmf.CMF(mf)
    cmf.set_clusters_from_ao([['0 H', '1 H'], ['2 H', '3 H']], [(1, 1), (1, 1)])
    cmf.algorithm = 'newton'
    cmf.kernel()
    return cmf


@pytest.fixture(scope='module')
def be2():
    return _be2()


def test_entropies_brute_force(be2):
    """<n_a>, <n_b>, <n_a n_b> and the entropies against determinant weights in the natural orbitals"""
    for i, ent in enumerate(be2.orbital_entropies()):
        norb, nelec = len(be2.clusters[i]), tuple(be2.fspace[i])
        ci = fci.addons.transform_ci_for_orbital_rotation(be2.ci[i], norb, nelec, ent['U'])
        da, db = fci.direct_spin1.make_rdm1s(ci, norb, nelec)
        assert np.allclose(da + db, np.diag(ent['occ']), atol=1e-10)       # the CI is in the natural orbitals
        P = ci**2
        sa, sb = cistring.make_strings(range(norb), nelec[0]), cistring.make_strings(range(norb), nelec[1])
        for p in range(norb):
            oa = (sa >> p) & 1
            ob = (sb >> p) & 1
            na, nb, nd = oa @ P.sum(1), P.sum(0) @ ob, oa @ P @ ob
            assert np.allclose([na, nb, nd], [ent['n_alpha'][p], ent['n_beta'][p], ent['n_double'][p]], atol=1e-10)
            w = np.array([1 - na - nb + nd, na - nd, nb - nd, nd])
            w = w[w > 1e-16]
            assert abs(-np.sum(w * np.log(w)) - ent['entropy'][p]) < 1e-8


def test_keep_all_is_noop(be2):
    """Keeping every orbital only rotates each cluster to its natural orbitals: same cMF energy"""
    cmf = _be2()
    e = cmf.e_tot
    sel = cmf.select_active_space(occ_thresh=0.0)
    assert sel['Cdoc_add'].shape[1] == 0 and sel['Cvir_add'].shape[1] == 0
    cmf.reduce_active_space_(sel)
    assert abs(cmf.cmf_ci() - e) < 1e-9


def test_reduce_folds_core_and_reoptimizes():
    """Be 1s (n ~ 2) becomes doubly occupied; electrons are conserved; the reduced ooCMF converges just above"""
    cmf = _be2()
    e_big, nel = cmf.e_tot, cmf.mol.nelectron
    sel = cmf.select_active_space(occ_thresh=0.01)
    assert [sum(r[2] == 'doc' for r in rows) for rows in sel['table']] == [1, 1]
    assert sel['fspace'] == [(1, 1), (1, 1)] and sel['Cdoc_add'].shape[1] == 2
    cmf.reduce_active_space_(sel)
    assert 2*cmf.Cdoc.shape[1] + sum(map(sum, cmf.fspace)) == nel
    C = cmf.get_mo_coeffs()
    assert np.allclose(C.T @ cmf.S @ C, np.eye(C.shape[1]), atol=1e-10)
    e = cmf.kernel()
    assert cmf.converged
    assert 0 <= e - e_big < 1e-3


def test_selection_from_pt1_rdm():
    """Natural orbitals and occupations from the cluster blocks of the PT1 many-body-expansion 1RDM"""
    cmf = _h2_dimer(2.0)
    e2, inc, dm1, dm1_ao = cmf.pt2_mbe(order=2, with_rdm=True)
    sel = cmf.select_active_space(occ_thresh=0.01, dm1=dm1)
    for i, ci in enumerate(cmf.clusters):
        occ = np.sort(np.linalg.eigvalsh(dm1[np.ix_(ci, ci)]))[::-1]
        assert np.allclose([r[0] for r in sel['table'][i]], occ)
        assert all(r[1] is None for r in sel['table'][i])
    assert sel['fspace'] == [(1, 1), (1, 1)]
    with pytest.raises(ValueError):
        cmf.select_active_space(occ_thresh=1.5)
