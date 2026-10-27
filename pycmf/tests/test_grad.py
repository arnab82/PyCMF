"""
Tests for the analytic nuclear gradients of orbital-optimized cMF
"""
import numpy as np
import pytest
from pyscf import gto, scf, mcscf

import pycmf
from pycmf.grad import _carry_orbitals


def _tight(cmf):
    cmf.conv_tol_grad = 1e-6
    cmf.conv_tol = 1e-11
    cmf.conv_tol_rdm1 = 1e-12
    cmf.ci_conv_tol = 1e-14
    return cmf


def _cmf(mol, labels, fspace, density_fit=False, rohf=False, sa=None, C0=None):
    mf = scf.ROHF(mol) if rohf else scf.RHF(mol)
    if density_fit:
        mf = mf.density_fit()
    mf.run(conv_tol=1e-12)
    cmf = _tight(pycmf.CMF(mf))
    cmf.set_clusters_from_ao(labels, fspace)
    if C0 is not None:
        cmf.set_mo_coeffs(_carry_orbitals(C0, cmf.S))
    if sa is not None:
        cmf.state_average_(sa)
    cmf.kernel()
    return cmf


def _finite_difference(cmf, coords, h=1e-4, **kwargs):
    """Central differences of the energy, each displaced geometry started from the orbitals of `cmf`
    """
    mol = cmf.mol
    xyz = mol.atom_coords()
    out = []
    for ia, x in coords:
        e = []
        for sign in (1, -1):
            c = xyz.copy()
            c[ia, x] += sign*h
            e.append(_cmf(mol.set_geom_(c, unit='Bohr', inplace=False), C0=cmf.get_mo_coeffs(), **kwargs).e_tot)
        out.append((e[0] - e[1]) / (2*h))
    return np.array(out)


@pytest.mark.parametrize('density_fit', [False, True])
def test_single_cluster_is_casscf(density_fit):
    """
    With one cluster cMF is CASSCF: same energy as CASCI in the optimized orbitals, and the CASSCF gradient there
    (exact and density-fitted integrals)
    """
    mol = gto.M(atom='N 0 0 0; N 0.1 0 1.15', basis='6-31g', verbose=0)
    mf = scf.RHF(mol)
    if density_fit:
        mf = mf.density_fit()
    mf.run(conv_tol=1e-12)
    C = mf.mo_coeff
    cmf = _tight(pycmf.CMF(mf))
    cmf.set_Cdoc(C[:, :4])
    cmf.set_Cact(C[:, 4:10], [list(range(6))], [(3, 3)])
    cmf.set_Cvir(C[:, 10:])
    cmf.kernel()
    mo = np.hstack((cmf.Cdoc, cmf.Cact, cmf.Cvir))
    mc = mcscf.CASCI(mf, 6, 6)
    mc.fcisolver.conv_tol = 1e-14
    mc.kernel(mo)
    assert abs(cmf.e_tot - mc.e_tot) < 1e-9
    g_ref = mcscf.CASSCF(mf, 6, 6).nuc_grad_method().kernel(mo_coeff=mo, ci=mc.ci)
    assert abs(cmf.nuc_grad() - g_ref).max() < 1e-5


def test_finite_difference():
    """H4 in two clusters (all electrons active)"""
    mol = gto.M(atom='H 0 0 0; H 0.1 0 0.9; H 0 1.6 0.2; H 0.2 1.7 1.1', basis='6-31g', verbose=0)
    kw = dict(labels=[['0 H', '1 H'], ['2 H', '3 H']], fspace=[(1, 1), (1, 1)])
    cmf = _cmf(mol, **kw)
    g = cmf.nuc_grad()
    coords = [(0, 2), (2, 1)]
    assert abs(np.array([g[c] for c in coords]) - _finite_difference(cmf, coords, **kw)).max() < 1e-5


def test_ro_cmf_density_fitting():
    """Two high-spin N atoms coupled antiferromagnetically (RO-cMF), with a core, density fitted"""
    mol = gto.M(atom='N 0 0 0; N 0.2 0.1 2.0', basis='6-31g', spin=6, verbose=0)
    kw = dict(labels=[['0 N 2p'], ['1 N 2p']], fspace=[(3, 0), (0, 3)], density_fit=True, rohf=True)
    cmf = _cmf(mol, **kw)
    g = cmf.nuc_grad()
    coords = [(1, 2)]
    assert abs(np.array([g[c] for c in coords]) - _finite_difference(cmf, coords, **kw)).max() < 1e-5
    assert abs(g.sum(axis=0)).max() < 1e-8         # translational invariance


def test_state_average():
    """Gradient of the averaged energy of the two charge localizations of (H2)2+"""
    mol = gto.M(atom='H 0 0 0; H 0 0 .74; H 2.5 0.1 0; H 2.6 0 .8', basis='6-31g', charge=1, spin=1, verbose=0)
    kw = dict(labels=[['0 H', '1 H'], ['2 H', '3 H']], fspace=[(1, 1), (1, 0)], rohf=True,
              sa=[[(1, 1), (1, 0)], [(1, 0), (1, 1)]])
    cmf = _cmf(mol, **kw)
    g = cmf.nuc_grad()
    coords = [(3, 2)]
    assert abs(np.array([g[c] for c in coords]) - _finite_difference(cmf, coords, **kw)).max() < 1e-5


def test_scanner():
    """The scanner rebuilds the clusters at a new geometry and gives the same energy and gradient"""
    mol = gto.M(atom='H 0 0 0; H 0.1 0 0.9; H 0 1.6 0.2; H 0.2 1.7 1.1', basis='6-31g', verbose=0)
    kw = dict(labels=[['0 H', '1 H'], ['2 H', '3 H']], fspace=[(1, 1), (1, 1)])
    cmf = _cmf(mol, **kw)
    scanner = cmf.nuc_grad_method().as_scanner()
    mol2 = mol.set_geom_(mol.atom_coords() + .05, unit='Bohr', inplace=False)
    e, g = scanner(mol2)
    ref = _cmf(mol2, **kw)
    assert scanner.converged
    assert abs(e - ref.e_tot) < 1e-8
    assert abs(g - ref.nuc_grad()).max() < 1e-5

    frz = pycmf.CMF(scf.RHF(gto.M(atom='N 0 0 0; N 0 0 1.1', basis='sto-3g', verbose=0)).run())
    frz.set_Cfrz(frz.mf.mo_coeff[:, :2])
    frz.set_Cdoc(frz.mf.mo_coeff[:, 2:4])
    frz.set_Cact(frz.mf.mo_coeff[:, 4:10], [list(range(6))], [(3, 3)])
    frz.set_Cvir(frz.mf.mo_coeff[:, 10:])
    frz.kernel()
    with pytest.raises(NotImplementedError):
        frz.nuc_grad()
