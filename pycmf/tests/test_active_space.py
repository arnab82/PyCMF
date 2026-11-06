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


def test_external_cluster_rdms(be2, tmp_path):
    """Block RDMs passed in (here the cMF ones, through the npz format of tps_pt1_cluster_rdms.jl) give the same
    occupations, entropies and selection as the cMF default; wrong sizes are rejected"""
    out = {}
    for i, ci in enumerate(be2.clusters):
        (a, b), (_, ab, _) = fci.direct_spin1.make_rdm12s(be2.ci[i], len(ci), tuple(be2.fspace[i]))
        for tag in ('', '0'):
            out['d1a%s_%d' % (tag, i+1)], out['d1b%s_%d' % (tag, i+1)], out['d2ab%s_%d' % (tag, i+1)] = a, b, ab
    np.savez(tmp_path / 'blocks.npz', **out)
    rdms = pycmf.active_space.load_cluster_rdms(tmp_path / 'blocks.npz')
    assert len(rdms) == len(be2.clusters)
    for x, y in zip(be2.orbital_entropies(), be2.orbital_entropies(rdms)):
        assert np.allclose(x['occ'], y['occ'], atol=1e-12) and np.allclose(x['entropy'], y['entropy'], atol=1e-12)
    s0, s1 = be2.select_active_space(occ_thresh=0.01), be2.select_active_space(occ_thresh=0.01, rdms=rdms)
    assert s0['fspace'] == s1['fspace'] and [[r[2] for r in t] for t in s0['table']] == [[r[2] for r in t] for t in s1['table']]
    with pytest.raises(ValueError):
        be2.orbital_entropies(rdms[:1])
    with pytest.raises(ValueError):
        be2.orbital_entropies([(np.eye(2), np.eye(2), np.zeros((2,)*4))]*2)
    with pytest.raises(ValueError):
        be2.select_active_space(dm1=np.eye(be2.Cact.shape[1]), rdms=rdms)


def test_roots_and_difference_orbitals(be2, tmp_path):
    """Root-suffixed blocks are read by root; the difference orbitals of two states diagonalize D_a - D_b"""
    rng = np.random.default_rng(3)
    out, blocks = {}, {}
    for r, suffix in ((1, ''), (2, '_r2')):
        for i, ci in enumerate(be2.clusters):
            n = len(ci)
            a = rng.standard_normal((n, n)); a = a @ a.T
            b = rng.standard_normal((n, n)); b = b @ b.T
            blocks[r, i] = (a, b)
            for tag in ('', '0'):
                out['d1a%s_%d%s' % (tag, i+1, suffix)], out['d1b%s_%d%s' % (tag, i+1, suffix)] = a, b
                out['d2ab%s_%d%s' % (tag, i+1, suffix)] = np.zeros((n,)*4)
    np.savez(tmp_path / 'roots.npz', **out)
    r1 = pycmf.active_space.load_cluster_rdms(tmp_path / 'roots.npz', root=1)
    r2 = pycmf.active_space.load_cluster_rdms(tmp_path / 'roots.npz', root=2)
    assert len(r1) == len(be2.clusters)
    for i in range(len(be2.clusters)):
        assert np.allclose(r2[i][0], blocks[2, i][0]) and np.allclose(r1[i][1], blocks[1, i][1])
    for i, x in enumerate(pycmf.active_space.difference_orbitals(be2, r2, r1)):
        dD = sum(blocks[2, i]) - sum(blocks[1, i])
        assert np.allclose(x['U'] @ np.diag(x['dn']) @ x['U'].T, dD)
        assert np.all(np.diff(abs(x['dn'])) <= 1e-12)
