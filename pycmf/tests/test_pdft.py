"""
Tests for pair-density functional theory on top of cMF
"""
import numpy as np
import pytest
from pyscf import gto, scf, mcpdft, fci

import pycmf
from pycmf import pdft


@pytest.mark.parametrize('otxc', ['tPBE', 'ftPBE'])
def test_single_cluster_is_mcpdft(otxc):
    """With one cluster, cMF-PDFT is MC-PDFT of CASCI over the same orbitals"""
    mol = gto.M(atom='N 0 0 0; N 0 0 1.15', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    C = mf.mo_coeff
    cmf = pycmf.CMF(mf)
    cmf.set_Cdoc(C[:, :4])
    cmf.set_Cact(C[:, 4:10], [list(range(6))], [(3, 3)])
    cmf.set_Cvir(C[:, 10:])
    cmf.ci_conv_tol = 1e-14
    cmf.conv_tol_rdm1 = 1e-12
    cmf.cmf_ci()
    mc = mcpdft.CASCI(mf, otxc, 6, 6, grids_level=2)
    mc.fcisolver.conv_tol = 1e-14
    mc.kernel(C)
    assert abs(pdft.energy(cmf, otxc, grids_level=2)[0] - mc.e_tot) < 1e-8


@pytest.fixture(scope='module')
def two_nitrogens():
    mol = gto.M(atom='N 0 0 0; N 0 0 2.5', basis='6-31g', spin=6, verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-12)
    cmf = pycmf.CMF(mf)
    cmf.set_clusters_from_ao([['0 N 2p'], ['1 N 2p']], [(3, 0), (0, 3)])
    cmf.kernel()
    return mf, cmf


def test_spin_coupled_rdms(two_nitrogens):
    """
    The 2RDMs of the spin-coupled cMF states (closed form) are those of the model-space states of QDPT2, and the
    high-spin one is the single determinant of CASCI with S = 3
    """
    mf, cmf = two_nitrogens
    res = cmf.qdpt2(with_rdm=True)
    for k in range(len(res.energies)):
        s = round(res.spins[k])
        corr = np.zeros((2, 2))
        corr[0, 1] = corr[1, 0] = pdft.pair_spin_correlation(s, 1.5, 1.5)
        dm1, dm2 = pdft.make_rdm12(cmf, corr)
        d1, d2 = res.make_rdm12(k, order=0)
        assert abs(dm2 - d2).max() < 1e-7
        assert abs(dm1 - d1).max() < 1e-7
    energies, j = pdft.spin_ladder(cmf, grids_level=2)
    mc = mcpdft.CASCI(mf, 'tPBE', 6, (6, 0), grids_level=2)
    mc.kernel(np.hstack((cmf.Cdoc, cmf.Cact, cmf.Cvir)))
    assert abs(energies[3.0] - mc.e_tot) < 1e-8


def test_exchange_coupling(two_nitrogens):
    """
    J from the first-order QDPT2 wave function is close to MC-PDFT on the CASCI spin states; the spin-coupled cMF
    states alone have no kinetic exchange and come out ferromagnetic
    """
    mf, cmf = two_nitrogens
    ot = pdft.make_ot(cmf.mol, 'tPBE', grids_level=2)
    res = cmf.qdpt2(with_rdm=True)
    e1, j1 = pdft.qdpt2_spin_ladder(cmf, res, order=1, ot=ot)
    e0, j0 = pdft.spin_ladder(cmf, ot=ot)
    assert sorted(e1) == [0, 1, 2, 3]
    C = np.hstack((cmf.Cdoc, cmf.Cact, cmf.Cvir))
    ecas = []
    for s in range(4):
        mc = mcpdft.CASCI(mf, 'tPBE', 6, (3, 3), grids_level=2)
        mc.fcisolver = fci.addons.fix_spin_(mc.fcisolver, ss=s*(s+1))
        mc.kernel(C)
        ecas.append(mc.e_tot)
    s = np.arange(4.)
    j_cas = np.linalg.lstsq(np.vstack((np.ones(4), -s*(s+1))).T, ecas, rcond=None)[0][1]
    assert j_cas < 0 and j1 < 0 < j0
    assert abs(j1 - j_cas) < .1*abs(j_cas)
