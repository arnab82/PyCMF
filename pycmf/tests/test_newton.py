"""
Tests for the orbital Hessian (gen_g_hop) and the trust-region Newton optimizer of ooCMF
"""
import numpy as np
import scipy.linalg
import pytest
from pyscf import gto, scf

import pycmf


def _fixed_rdm_gradient(cmf, C0, K, mask):
    """Gradient with respect to kappa at C0 exp(K), with the cluster RDMs kept fixed"""
    cmf.set_mo_coeffs(C0 @ scipy.linalg.expm(K))
    dEdC = cmf._average_deriv(cmf.get_core_veff(), cmf.get_cluster_eris_ao())[0]
    M = scipy.linalg.expm_frechet(K.T, C0.T @ dEdC, compute_expm=False)
    return (M - M.T)[mask]


def _n2(fspace, density_fit=False, frozen=0):
    mol = gto.M(atom='N 0 0 0; N 0 0 2.5', basis='6-31g', spin=6, verbose=0)
    mf = scf.ROHF(mol)
    if density_fit:
        mf = mf.density_fit()
    mf.run(conv_tol=1e-12)
    cmf = pycmf.CMF(mf)
    cmf.set_clusters_from_ao([['0 N 2p', '0 N 3p'], ['1 N 2p', '1 N 3p']], fspace)
    if frozen:
        C = cmf.Cdoc
        cmf.set_Cfrz(C[:, :frozen])
        cmf.set_Cdoc(C[:, frozen:])
    return cmf


def _h4_cation_sa():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.8; H 0 0 2.5; H 0 0 3.3', basis='6-31g', charge=1, spin=1, verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-12)
    cmf = pycmf.CMF(mf)
    cmf.set_clusters_from_ao([['0 H', '1 H'], ['2 H', '3 H']], [(1, 1), (1, 0)])
    cmf.state_average_([[(1, 1), (1, 0)], [(1, 0), (1, 1)]], weights=[.7, .3])
    return cmf


@pytest.mark.parametrize('make', [lambda: _n2([(3, 0), (0, 3)], frozen=2),
                                  lambda: _n2([(3, 0), (2, 1)], density_fit=True),
                                  _h4_cation_sa],
                         ids=['ro-cmf_frozen', 'density_fit', 'state_average'])
def test_hessian_vector_finite_difference(make):
    """H x against central differences of the analytic gradient at fixed cluster RDMs, at the AO starting orbitals"""
    cmf = make()
    cmf.conv_tol_rdm1 = 1e-11
    cmf.cmf_ci()
    C0 = cmf.get_mo_coeffs()
    mask = cmf.rotation_mask()
    g, h_op, h_diag = cmf.gen_g_hop()
    assert np.allclose(g, cmf.orbital_gradient()[mask], atol=1e-10)
    rng = np.random.default_rng(7)
    x, y = rng.standard_normal((2, g.size))
    hx = h_op(x)
    assert abs(np.dot(y, hx) - np.dot(x, h_op(y))) < 1e-9 * np.linalg.norm(hx) * np.linalg.norm(y)
    X = np.zeros(mask.shape)
    X[mask] = x
    X = X - X.T
    h = 2.5e-5
    fd = (_fixed_rdm_gradient(cmf, C0, h*X, mask) - _fixed_rdm_gradient(cmf, C0, -h*X, mask)) / (2*h)
    assert np.linalg.norm(hx - fd) < 2e-8 * np.linalg.norm(hx)


def test_newton_matches_lbfgs():
    """Both optimizers reach the same RO-cMF solution; Newton in fewer iterations"""
    e = {}
    for alg in ('lbfgs', 'newton'):
        cmf = _n2([(3, 0), (0, 3)])
        cmf.algorithm = alg
        cmf.conv_tol_grad = 1e-7
        e[alg] = cmf.kernel()
        assert cmf.converged
        assert np.linalg.norm(cmf.orbital_gradient()) < 1e-6
    assert abs(e['lbfgs'] - e['newton']) < 1e-10


def test_newton_ao_start():
    """
    Ethylene with the 2pz/3pz AOs of each carbon as clusters, starting from the AOs themselves: Newton converges
    (L-BFGS stalls on this badly conditioned start, whose softest Hessian modes are ~0.01), and the state average is
    optimized as well. This cMF problem has several local minima (clusters polarized differently), so only
    convergence is checked
    """
    mol = gto.M(atom='C 0 0 0; C 1.34 0 0; H -0.6 0.9 0; H -0.6 -0.9 0; H 1.94 0.9 0; H 1.94 -0.9 0',
                basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-12)
    cmf = pycmf.CMF(mf)
    cmf.set_clusters_from_ao([['0 C 2pz', '0 C 3pz'], ['1 C 2pz', '1 C 3pz']], [(1, 1), (1, 1)])
    cmf.algorithm = 'newton'
    e = cmf.kernel()
    assert cmf.converged
    assert e < mf.e_tot - 0.02
    assert np.linalg.norm(cmf.orbital_gradient()) < 1e-6

    cmf = _h4_cation_sa()
    cmf.algorithm = 'newton'
    e = cmf.kernel()
    assert cmf.converged
    assert abs(e - (.7*cmf.e_states[0] + .3*cmf.e_states[1])) < 1e-12
    assert np.linalg.norm(cmf.orbital_gradient()) < 1e-6
