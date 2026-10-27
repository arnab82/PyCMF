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
