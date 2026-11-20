"""
Linearized pair-density functional theory (L-PDFT, Hennefarth et al., JCTC 2023) from reduced density matrices, for
multistate PDFT on wavefunctions that come as RDMs rather than CI vectors (TPSCI, cluster methods).

L-PDFT expands the MC-PDFT energy to first order around the state-averaged densities (D0, G0) of the model states,
which gives an effective Hamiltonian H^L = c + sum_pq h_pq E_pq + 1/2 sum_pqrs v_pqrs e_pqrs in the active space, and

    H^L_IJ = <I|H^L|J> + delta_IJ lambda E^wfn_I

(lambda: wavefunction fraction of hybrid on-top functionals). Diagonalizing H^L gives the L-PDFT energies and the
states as mixtures of the model states, which also defines a smooth multistate treatment near crossings. Here
<I|H^L|J> = sum_pq h_pq D^IJ_pq + 1/2 sum_pqrs v_pqrs G^IJ_pqrs with the state (I = J) and transition (I != J) RDMs, so
any method that provides them can be used; the effective integrals h, v, c are those of pyscf-forge (pyscf.mcpdft.lpdft),
computed from the averaged RDMs.
"""
import numpy as np
from pyscf import mcscf, mcpdft


def lpdft_hamiltonian(mf, mo_coeff, ncore, ncas, nelecas, dm1, dm2, tdm1=None, tdm2=None, e_wfn=None, otxc='tPBE',
                      weights=None, grids_level=None, verbose=0):
    """
    L-PDFT Hamiltonian of R model states from their RDMs

    Parameters
    ----------
    mf : SCF object (molecule, integrals, density fitting)
    mo_coeff : (nao, nmo) orbitals ordered [core, active, virtual]
    ncore, ncas, nelecas : core and active orbitals, active electrons (na, nb)
    dm1 : (R, ncas, ncas) spin-summed 1RDMs of the model states
    dm2 : (R, ncas, ncas, ncas, ncas) spin-summed 2RDMs, PySCF convention dm2[p,q,r,s] = <p' r' s q>
    tdm1, tdm2 : dict (I, J) -> transition RDMs <I|...|J> (I < J), same conventions; symmetrized here, as H^L is
        real symmetric. Missing pairs (e.g. different spins) are zero
    e_wfn : wavefunction energies of the model states, for hybrid functionals: None (default) computes <I|H|J> from
        the RDMs with the integrals of `mf`; (R,) replaces the diagonal (e.g. energies with exact integrals); (R, R)
        is used as the whole wavefunction Hamiltonian
    otxc : on-top functional
    weights : (R,) weights of the expansion point (default: equal, as the L-PDFT gradients require)

    The spin-separated 1RDMs are taken as dm1/2 for both spins: exact for M_s = 0 states of any S (the spin density of
    a pure spin state is proportional to M_s), and translated functionals only use the spin-summed densities anyway.

    Returns
    -------
    H : (R, R) L-PDFT Hamiltonian
    e : (R,) L-PDFT energies (eigenvalues, ascending)
    U : (R, R) eigenvectors (columns): the L-PDFT states as combinations of the model states
    """
    dm1, dm2 = np.asarray(dm1), np.asarray(dm2)
    R = dm1.shape[0]
    w = np.full(R, 1 / R) if weights is None else np.asarray(weights, dtype=float)
    mc = mcpdft.CASCI(mf, otxc, ncas, nelecas)
    mc.mo_coeff = mo_coeff
    mc.ncore = ncore
    mc.verbose = verbose
    if grids_level is not None:
        mc.grids.level = grids_level
    lm = mc.multi_state(list(w), method='lin')
    ot = lm.otfnal
    ot.reset(mol=mf.mol)
    spin = abs(nelecas[0] - nelecas[1])
    omega, _, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)
    if abs(omega) > 1e-11 or abs(hyb[0] - hyb[1]) > 1e-11:
        raise NotImplementedError('range-separated or unequal-hybrid on-top functionals')
    lam = hyb[0]

    D0 = np.einsum('k,kpq->pq', w, dm1)
    G0 = np.einsum('k,kpqrs->pqrs', w, dm2)
    casdm1s_0 = np.array([.5 * D0, .5 * D0])
    lm.veff1, lm.veff2, E_ot = lm.get_pdft_veff(mo=mo_coeff, casdm1s=casdm1s_0, casdm2=G0, drop_mcwfn=True,
                                                incl_energy=True, ot=ot)
    h1, h0 = lm.get_h1lpdft(E_ot, casdm1s_0, G0, hyb=1.0 - lam, mo_coeff=mo_coeff)
    h2 = lm.get_h2lpdft()

    pairs = {}
    for (i, j), t1 in (tdm1 or {}).items():
        t2 = tdm2[(i, j)]
        pairs[(i, j)] = (.5 * (t1 + t1.T), .5 * (t2 + t2.transpose(1, 0, 3, 2)))

    def contract(c, g1, g2, d1, d2):
        return c + np.einsum('pq,pq', g1, d1) + .5 * np.einsum('pqrs,pqrs', g2, d2)

    H = np.zeros((R, R))
    for i in range(R):
        H[i, i] = contract(h0, h1, h2, dm1[i], dm2[i])
    for (i, j), (t1, t2) in pairs.items():
        H[i, j] = H[j, i] = contract(0, h1, h2, t1, t2)
    if lam:
        H += lam * wavefunction_hamiltonian(mf, mo_coeff, ncore, ncas, dm1, dm2, pairs, e_wfn)
    e, U = np.linalg.eigh(H)
    return H, e, U


def wavefunction_hamiltonian(mf, mo_coeff, ncore, ncas, dm1, dm2, pairs, e_wfn=None):
    """
    <I|H|J> of the model states from their RDMs and the active-space Hamiltonian of `mf` (core energy, dressed
    one-electron and two-electron integrals); `pairs` (I, J) -> symmetrized transition RDMs. With `e_wfn` given as
    (R,) energies they replace the diagonal (e.g. TPSCI energies with exact integrals), as (R, R) the whole matrix.
    """
    if e_wfn is not None and np.ndim(e_wfn) == 2:
        return np.asarray(e_wfn, dtype=float)
    from pyscf import ao2mo
    cas = mcscf.CASCI(mf, ncas, (1, 1))
    cas.ncore = ncore
    h1e, ecore = cas.get_h1eff(mo_coeff)
    eri = ao2mo.restore(1, cas.get_h2eff(mo_coeff), ncas)
    R = len(dm1)
    W = np.zeros((R, R))
    for i in range(R):
        W[i, i] = ecore + np.einsum('pq,pq', h1e, dm1[i]) + .5 * np.einsum('pqrs,pqrs', eri, dm2[i])
    for (i, j), (t1, t2) in pairs.items():
        W[i, j] = W[j, i] = np.einsum('pq,pq', h1e, t1) + .5 * np.einsum('pqrs,pqrs', eri, t2)
    if e_wfn is not None:
        W[np.diag_indices(R)] = np.asarray(e_wfn, dtype=float)
    return W


def lpdft_from_cmf(cmf, dm1, dm2, tdm1=None, tdm2=None, e_wfn=None, otxc='tPBE', weights=None, grids_level=None,
                   verbose=0):
    """`lpdft_hamiltonian` with the orbitals of a CMF object: core = frozen + doubly occupied, active = all clusters in
    cluster order (the order of `CMF.export_fermicg` and of TPSCI RDMs), virtual"""
    C = np.hstack((cmf.Cfrz, cmf.Cdoc, cmf.Cact, cmf.Cvir))
    ncore = cmf.Cfrz.shape[1] + cmf.Cdoc.shape[1]
    ncas = cmf.Cact.shape[1]
    na = sum(f[0] for f in cmf.fspace)
    nb = sum(f[1] for f in cmf.fspace)
    return lpdft_hamiltonian(cmf.mf, C, ncore, ncas, (na, nb), dm1, dm2, tdm1, tdm2, e_wfn, otxc, weights,
                             grids_level, verbose)
