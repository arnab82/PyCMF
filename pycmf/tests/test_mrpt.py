"""
Tests for SC-NEVPT2 on top of cMF
"""
import numpy as np
import pytest
from pyscf import gto, scf, mcscf, mrpt

import pycmf


def _single_cluster(mf, ncore, ncas, nelecas):
    C = mf.mo_coeff
    cmf = pycmf.CMF(mf)
    cmf.set_Cdoc(C[:, :ncore])
    cmf.set_Cact(C[:, ncore:ncore+ncas], [list(range(ncas))], [nelecas])
    cmf.set_Cvir(C[:, ncore+ncas:])
    cmf.conv_tol_rdm1 = 1e-12
    cmf.ci_conv_tol = 1e-14
    cmf.cmf_ci()
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.fcisolver.conv_tol = 1e-14
    mc.kernel(C)
    return cmf, mrpt.NEVPT(mc).kernel()


def test_single_cluster_is_nevpt2():
    """One cluster: the cMF state is the CASCI state and cMF-NEVPT2 is SC-NEVPT2 (closed and open shell)"""
    mol = gto.M(atom='N 0 0 0; N 0 0 1.3', basis='6-31g', verbose=0)
    cmf, e_ref = _single_cluster(scf.RHF(mol).run(conv_tol=1e-12), 4, 6, (3, 3))
    assert abs(cmf.nevpt2()[0] - e_ref) < 1e-6

    mol = gto.M(atom='O 0 0 0; O 0 0 1.21', basis='6-31g', spin=2, verbose=0)
    cmf, e_ref = _single_cluster(scf.ROHF(mol).run(conv_tol=1e-12), 4, 6, (5, 3))
    assert abs(cmf.nevpt2(spin_average=False)[0] - e_ref) < 1e-6
    assert abs(cmf.nevpt2(spin_average=True)[0] - e_ref) < 2e-6


def test_density_fitting():
    """Density fitting: the core-core/virtual-virtual class from 3-index integrals, against DF-NEVPT2"""
    mol = gto.M(atom='N 0 0 0; N 0 0 1.3', basis='6-31g', verbose=0)
    cmf, e_ref = _single_cluster(scf.RHF(mol).density_fit().run(conv_tol=1e-12), 4, 6, (3, 3))
    assert abs(cmf.nevpt2()[0] - e_ref) < 1e-6


def test_size_consistency():
    """Two N2 molecules 50 A apart, one cluster each: twice the NEVPT2 energy of one molecule"""
    mol1 = gto.M(atom='N 0 0 0; N 0 0 1.3', basis='6-31g', verbose=0)
    mf1 = scf.RHF(mol1).run(conv_tol=1e-12)
    perm = [0, 1, 2, 3, 5, 6, 7, 8, 4] + list(range(9, mf1.mo_coeff.shape[1]))
    C1 = mf1.mo_coeff[:, perm]
    mc1 = mcscf.CASCI(mf1, 4, 4)
    mc1.fcisolver.conv_tol = 1e-14
    mc1.kernel(C1)
    e1 = mrpt.NEVPT(mc1).kernel()

    mol2 = gto.M(atom='N 0 0 0; N 0 0 1.3; N 50 0 0; N 50 0 1.3', basis='6-31g', verbose=0)
    mf2 = scf.RHF(mol2)
    mf2.kernel(dm0=np.kron(np.eye(2), mf1.make_rdm1()))
    Z = np.zeros_like(C1)
    CA, CB = np.vstack((C1, Z)), np.vstack((Z, C1))
    cmf = pycmf.CMF(mf2)
    cmf.set_Cfrz(np.hstack((CA[:, :1], CB[:, :1])))
    cmf.set_Cdoc(np.hstack((CA[:, 1:5], CB[:, 1:5])))
    cmf.set_Cact(np.hstack((CA[:, 5:9], CB[:, 5:9])), [[0, 1, 2, 3], [4, 5, 6, 7]], [(2, 2), (2, 2)])
    cmf.set_Cvir(np.hstack((CA[:, 9:], CB[:, 9:])))
    cmf.conv_tol_rdm1 = 1e-12
    cmf.ci_conv_tol = 1e-14
    cmf.cmf_ci()
    e2, parts = cmf.nevpt2()

    # the same with the 1s orbitals frozen in the single molecule: freeze them in the reference too
    cmf1 = pycmf.CMF(mf1)
    cmf1.set_Cfrz(C1[:, :1])
    cmf1.set_Cdoc(C1[:, 1:5])
    cmf1.set_Cact(C1[:, 5:9], [[0, 1, 2, 3]], [(2, 2)])
    cmf1.set_Cvir(C1[:, 9:])
    cmf1.conv_tol_rdm1 = 1e-12
    cmf1.ci_conv_tol = 1e-14
    cmf1.cmf_ci()
    e1_frz = cmf1.nevpt2()[0]
    assert abs(e2 - 2*e1_frz) < 1e-8
    assert e1 < e1_frz < 0
