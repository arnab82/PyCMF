"""
Pair-density functional theory on top of cMF (cMF-PDFT)

    E = V_nn + tr(D h) + 1/2 tr(D J[D]) + E_ot[rho, Pi]

as in MC-PDFT, with the density and on-top pair density Pi of a cMF state and the on-top functionals of
`pyscf.mcpdft` (translated 'tPBE', 'tBLYP', ..., fully translated 'ftPBE', ...). These functionals depend on the
total density and Pi only.

A product of cluster states has a 2RDM that is mean field between the clusters. The spin coupling of open-shell
clusters, over which the RO-cMF energy averages, enters through the exchange part between two clusters: for cluster
ground multiplets of spin S_I, by the Wigner-Eckart theorem,

    G[p,q,r,s] = -1/2 g^I_ps g^J_rq - 2 <S_I.S_J> m^I_ps m^J_rq / (S_I S_J)        (p,s in I; r,q in J)

with g^I the spin-summed 1RDM of cluster I and m^I = (g^I_alpha - g^I_beta)/2 the spin density matrix of its
M_I = S_I component. <S_I.S_J> = 0 is the RO-cMF ensemble (the 2RDM of `CMF.make_rdm12`), and two clusters coupled to
total spin S have <S_A.S_B> = [S(S+1) - S_A(S_A+1) - S_B(S_B+1)]/2. All spin states have the same density and
classical energy, and differ by E_ot through Pi, which gives the exchange coupling J (`spin_ladder`). No determinant
space is needed, so the active space can be as large as cMF allows.

Correlated wave functions over a few clusters (the QDPT2 first-order wave function, with charge transfer between the
clusters) enter through their RDMs, `energy_qdpt2`.
"""
import numpy as np
from pyscf.lib import logger
from pyscf.mcpdft import otfnal


def make_ot(mol, otxc='tPBE', grids_level=None):
    """On-top functional object with its integration grid built
    """
    ot = otfnal.get_transfnal(mol, otxc)
    if np.abs(ot._numint.rsh_and_hybrid_coeff(ot.otxc, mol.spin)[2]).max() > 1e-10:
        raise NotImplementedError('hybrid on-top functionals')
    if grids_level is not None:
        ot.grids.level = grids_level
    ot.grids.build()
    return ot


def cluster_spins(cmf) -> list:
    """(S_I, m^I) of each cluster: spin and spin density matrix (cluster orbitals) of the M_I = S_I component
    """
    from pyscf import fci
    from .pt2 import _multiplet
    out = []
    for i, ci in enumerate(cmf.clusters):
        n, nelec = len(ci), tuple(cmf.fspace[i])
        ss = fci.spin_square(cmf.ci[i], n, nelec)[0]
        s = .5*(np.sqrt(1 + 4*max(ss, 0)) - 1)
        if s < 1e-4:
            out.append((0.0, np.zeros((n, n))))
            continue
        comps = _multiplet(cmf.ci[i], n, nelec)
        sec = max(comps, key=lambda x: x[0] - x[1])
        da, db = fci.direct_spin1.make_rdm1s(comps[sec], n, sec)
        out.append((s, .5*(da - db)))
    return out


def pair_spin_correlation(s, s_a, s_b) -> float:
    """<S_A.S_B> of two spins s_a, s_b coupled to total spin s
    """
    return .5*(s*(s+1) - s_a*(s_a+1) - s_b*(s_b+1))


def make_rdm12(cmf, spin_corr=None, spins=None):
    """
    Spin-summed 1RDM and spin-free 2RDM of the active orbitals for the cMF state with spin correlations
    <S_I.S_J> = spin_corr[I][J] between the clusters (default: 0, the RO-cMF ensemble, consistent with `e_tot`)
    """
    dm1, dm2 = cmf.make_rdm12()
    if spin_corr is None:
        return dm1, dm2
    spin_corr = np.asarray(spin_corr, dtype=float)
    if spins is None:
        spins = cluster_spins(cmf)
    nact = dm1.shape[0]
    blocks = []
    for (s, m), ci in zip(spins, cmf.clusters):
        b = np.zeros((nact, nact))
        b[np.ix_(ci, ci)] = m
        blocks.append((s, b))
    for i in range(len(cmf.clusters)):
        for j in range(i+1, len(cmf.clusters)):
            x = spin_corr[i, j]
            (si, mi), (sj, mj) = blocks[i], blocks[j]
            if abs(x) < 1e-14 or si == 0 or sj == 0:
                continue
            c = 2*x / (si*sj)
            dm2 -= c*(np.einsum('ps,rq->pqrs', mi, mj) + np.einsum('ps,rq->pqrs', mj, mi))
    return dm1, dm2


def _classical(cmf, dm1):
    """V_nn + tr(D h) + 1/2 tr(D J[D]) and the active orbitals, for the active-space 1RDM dm1
    """
    mf = cmf.mf
    Ccore = np.hstack((cmf.Cfrz, cmf.Cdoc))
    D = 2*Ccore @ Ccore.T + cmf.Cact @ dm1 @ cmf.Cact.T
    vj = mf.get_j(cmf.mol, D)
    return mf.energy_nuc() + np.einsum('ij,ji->', D, mf.get_hcore()) + .5*np.einsum('ij,ji->', D, vj)


def energy_ot(cmf, ot, dm1, dm2, max_memory=2000) -> float:
    """On-top energy E_ot[rho, Pi] of the active-space RDMs dm1 (spin summed) and dm2 (spin free)
    """
    Ccore = np.hstack((cmf.Cfrz, cmf.Cdoc))
    C = np.hstack((Ccore, cmf.Cact))
    casdm1s = np.array((.5*dm1, .5*dm1))
    return otfnal.energy_ot(ot, casdm1s, dm2, C, Ccore.shape[1], max_memory=max_memory)


def energy(cmf, otxc='tPBE', spin_corr=None, rdm12=None, ot=None, grids_level=None, verbose=None) -> (float, float):
    """
    cMF-PDFT energy

    Parameters
    ----------
    otxc : str
        On-top functional, e.g. 'tPBE', 'ftPBE', 'tBLYP' (ignored if `ot` is given)
    spin_corr : np.ndarray
        <S_I.S_J> between the clusters, see `make_rdm12` (default: the RO-cMF ensemble)
    rdm12 : tuple
        Active-space (dm1, dm2) of another state instead
    ot : otfnal
        On-top functional object (`make_ot`), to reuse its grid

    Returns
    -------
    e_tot : float
    e_ot : float
        On-top part
    """
    log = logger.new_logger(cmf, verbose)
    if ot is None:
        ot = make_ot(cmf.mol, otxc, grids_level)
    dm1, dm2 = make_rdm12(cmf, spin_corr) if rdm12 is None else rdm12
    e_ot = energy_ot(cmf, ot, dm1, dm2, getattr(cmf.mf, 'max_memory', 2000))
    e_tot = _classical(cmf, dm1) + e_ot
    log.info('cMF-PDFT (%s): E = %.12f  E_ot = %.12f', ot.otxc, e_tot, e_ot)
    return e_tot, e_ot


def spin_ladder(cmf, pair=None, otxc='tPBE', ot=None, grids_level=None, verbose=None) -> (dict, float):
    """
    cMF-PDFT energies of two open-shell clusters coupled to each total spin S (the other clusters in their cMF
    states), and the exchange coupling J of H = -2J S_A.S_B fitted to them, E(S) = E0 - J S(S+1)
    (positive J: ferromagnetic)

    Parameters
    ----------
    pair : tuple
        The two clusters (default: the two open-shell clusters)

    Returns
    -------
    energies : dict
        S -> E(S)
    j : float
        J (Hartree)
    """
    log = logger.new_logger(cmf, verbose)
    spins = cluster_spins(cmf)
    if pair is None:
        pair = [i for i, (s, m) in enumerate(spins) if s > 1e-4]
        if len(pair) != 2:
            raise ValueError('give the pair of open-shell clusters')
    a, b = pair
    sa, sb = spins[a][0], spins[b][0]
    if ot is None:
        ot = make_ot(cmf.mol, otxc, grids_level)
    dm1 = cmf.make_rdm1()
    e_cl = _classical(cmf, dm1)
    n = len(cmf.clusters)
    energies = {}
    for k in range(int(round(2*min(sa, sb))) + 1):
        s = abs(sa - sb) + k
        corr = np.zeros((n, n))
        corr[a, b] = corr[b, a] = pair_spin_correlation(s, sa, sb)
        dm2 = make_rdm12(cmf, corr, spins)[1]
        energies[s] = e_cl + energy_ot(cmf, ot, dm1, dm2)
        log.info('cMF-PDFT (%s)  S = %.1f  E = %.12f', ot.otxc, s, energies[s])
    s = np.array(sorted(energies))
    e = np.array([energies[x] for x in s])
    j = np.linalg.lstsq(np.vstack((np.ones_like(s), -s*(s+1))).T, e, rcond=None)[0][1]
    log.note('cMF-PDFT (%s) exchange coupling J = %.3f cm-1  (H = -2J S_A.S_B)', ot.otxc, j*219474.6313705)
    return energies, j


def embed_rdm12(cmf, orbs, dm1_sub, dm2_sub):
    """
    Active-space RDMs of a state of the clusters with orbitals `orbs` (in this order) times the cMF states (RO-cMF
    ensembles) of the other clusters: mean field between the two parts
    """
    nact = cmf.Cact.shape[1]
    orbs = np.asarray(orbs)
    dm1 = cmf.make_rdm1()
    dm1[np.ix_(orbs, orbs)] = dm1_sub
    dm2 = np.einsum('pq,rs->pqrs', dm1, dm1) - .5*np.einsum('ps,rq->pqrs', dm1, dm1)
    inside = set(orbs.tolist())
    for ci, d2 in zip(cmf.clusters, cmf.cluster_2rdm):
        if not inside.intersection(ci):
            dm2[np.ix_(ci, ci, ci, ci)] = d2
    dm2[np.ix_(orbs, orbs, orbs, orbs)] = dm2_sub
    return dm1, dm2


def energy_qdpt2(cmf, res, state=0, order=1, otxc='tPBE', ot=None, grids_level=None, verbose=None):
    """
    cMF-PDFT energy of a QDPT2 state (`qdpt2(..., with_rdm=True)`): its first-order (order=1) or model-space (order=0)
    wave function for the selected clusters, the other clusters in their cMF states
    """
    dm1_sub, dm2_sub = res.make_rdm12(state, order)
    return energy(cmf, otxc, rdm12=embed_rdm12(cmf, res.orbs, dm1_sub, dm2_sub), ot=ot, grids_level=grids_level,
                  verbose=verbose)


def qdpt2_spin_ladder(cmf, res, order=1, otxc='tPBE', ot=None, grids_level=None, verbose=None) -> (dict, float):
    """
    cMF-PDFT energies of the QDPT2 states (`qdpt2(..., with_rdm=True)`), the lowest one of each total spin, and J of
    H = -2J S_A.S_B fitted to them. With order=1 the first-order wave function includes the charge-transfer
    configurations that carry kinetic exchange; the spin-coupled cMF states alone (order=0, or `spin_ladder`) have
    none and miss the antiferromagnetic part.

    Returns
    -------
    energies : dict
        S -> E(S)
    j : float
        J (Hartree)
    """
    log = logger.new_logger(cmf, verbose)
    if ot is None:
        ot = make_ot(cmf.mol, otxc, grids_level)
    lowest = {}
    for k in np.argsort(res.energies):
        lowest.setdefault(round(2*res.spins[k]) / 2, k)
    energies = {}
    for s, k in sorted(lowest.items()):
        energies[s] = energy_qdpt2(cmf, res, k, order, ot=ot)[0]
        log.info('cMF-PDFT (%s) QDPT2 state %d  S = %.1f  E = %.12f', ot.otxc, k, s, energies[s])
    s = np.array(sorted(energies))
    e = np.array([energies[x] for x in s])
    j = np.linalg.lstsq(np.vstack((np.ones_like(s), -s*(s+1))).T, e, rcond=None)[0][1]
    log.note('cMF-PDFT (%s, QDPT2 order %d) exchange coupling J = %.3f cm-1  (H = -2J S_A.S_B)', ot.otxc, order,
             j*219474.6313705)
    return energies, j


def spin_square_rdm(dm1, dm2) -> float:
    """<S^2> from the spin-summed 1RDM and the spin-free 2RDM (PySCF convention) of an active space
    """
    return .75*np.trace(dm1) - .5*np.einsum('pqqp', dm2) - .25*np.einsum('ppqq', dm2)


def load_tpsci_rdms(filename):
    """
    Energies and RDMs of TPSCI states written by `examples/tpsci_rdms.jl` (TPSChem.jl): (R,), spin-summed 1RDMs
    (R, n, n) and spin-free 2RDMs (R, n, n, n, n) converted to the PySCF convention, dm2[p,q,r,s] = <p+ r+ s q>
    (TPSChem: G[p,q,r,s] = <p+ q+ s r>), in the orbital order of `CMF.export_fermicg`
    """
    d = np.load(filename)
    e = np.asarray(d['energies'], dtype=float).ravel()
    dm1 = np.moveaxis(np.asarray(d['dm1'], dtype=float), -1, 0)
    dm2 = np.moveaxis(np.asarray(d['dm2'], dtype=float), -1, 0).transpose(0, 1, 3, 2, 4)
    return e, dm1, dm2


def tpsci_spin_ladder(cmf, filename, clusters=None, otxc='tPBE', ot=None, grids_level=None, verbose=None):
    """
    PDFT on TPSCI states: the RDMs of each root from `examples/tpsci_rdms.jl` (TPSChem.jl), run on
    `cmf.export_fermicg(file, clusters)` with the same clusters, embedded in the cMF states of the other clusters

    Returns
    -------
    states : list
        (E_TPSCI, S, E_PDFT) of each root, S from <S^2> of its RDMs
    j_tpsci, j_pdft : float
        J of H = -2J S_A.S_B (Hartree) from the lowest root of each total spin, TPSCI and TPSCI-PDFT energies
    """
    log = logger.new_logger(cmf, verbose)
    e_ci, dm1s, dm2s = load_tpsci_rdms(filename)
    if clusters is None:
        clusters = range(len(cmf.clusters))
    orbs = np.concatenate([cmf.clusters[i] for i in clusters])
    if dm1s.shape[1] != len(orbs):
        raise ValueError('the RDMs have %d orbitals, the clusters %d' % (dm1s.shape[1], len(orbs)))
    if ot is None:
        ot = make_ot(cmf.mol, otxc, grids_level)
    states = []
    for r in range(len(e_ci)):
        ss = spin_square_rdm(dm1s[r], dm2s[r])
        s = round(np.sqrt(1 + 4*max(ss, 0)) - 1) / 2
        e = energy(cmf, rdm12=embed_rdm12(cmf, orbs, dm1s[r], dm2s[r]), ot=ot)[0]
        states.append((e_ci[r], s, e))
        log.info('TPSCI root %d  E = %.10f  <S^2> = %.4f  E(%s) = %.10f', r, e_ci[r], ss, ot.otxc, e)
    lowest = {}
    for e_r, s, e_p in sorted(states):
        lowest.setdefault(s, (e_r, e_p))
    s = np.array(sorted(lowest))
    a = np.vstack((np.ones_like(s), -s*(s+1))).T
    j_ci = np.linalg.lstsq(a, [lowest[x][0] for x in s], rcond=None)[0][1]
    j_pdft = np.linalg.lstsq(a, [lowest[x][1] for x in s], rcond=None)[0][1]
    log.note('TPSCI J = %.3f cm-1   TPSCI-PDFT (%s) J = %.3f cm-1', j_ci*219474.6313705, ot.otxc, j_pdft*219474.6313705)
    return states, j_ci, j_pdft


def mbe_rdm12(cmf, subsets, order=1, verbose=None, **kwargs) -> dict:
    """
    Active-space RDMs of the spin states from a many-body expansion over sets of clusters, e.g. the two magnetic
    clusters and each magnetic-bridge-magnetic triple: for each total spin S, the RDMs of the lowest QDPT2 state of
    spin S in each set (first-order wave function for order=1, embedded in the cMF states of the other clusters)
    are combined as the sum of the increments RDM_T - (increments of the subsets of T in `subsets`). What this drops
    are products of corrections on different sets (e.g. charge transfer on two bridges at once).

    Parameters
    ----------
    subsets : list
        Sets of clusters, each containing the magnetic clusters (and all its subsets in the list), e.g.
        [(0, 1), (0, 1, 2), (0, 1, 3), (0, 1, 4)]
    kwargs
        Passed to `qdpt2`

    Returns
    -------
    dict
        S -> (dm1, dm2), spin-summed 1RDM and spin-free 2RDM of all active orbitals
    """
    from .pt2 import mbe_increments
    log = logger.new_logger(cmf, verbose)
    subsets = [tuple(sorted(x)) for x in subsets]
    eri_act = cmf.get_active_eris()
    vcore = cmf.get_core_veff()
    rdms = {}
    for sub in subsets:
        res = cmf.qdpt2(list(sub), eri_act=eri_act, vcore=vcore, with_rdm=True, **kwargs)
        lowest = {}
        for k in np.argsort(res.energies):
            lowest.setdefault(round(2*res.spins[k]) / 2, k)
        rdms[sub] = {s: embed_rdm12(cmf, res.orbs, *res.make_rdm12(k, order)) for s, k in lowest.items()}
        log.info('clusters %s: J(QDPT2) = %.3f cm-1', sub, res.heisenberg_j()*219474.6313705)
    out = {}
    for s in sorted(set.intersection(*[set(v) for v in rdms.values()])):
        out[s] = tuple(sum(mbe_increments({t: rdms[t][s][x] for t in subsets}, subsets).values()) for x in (0, 1))
    return out


def mbe_spin_ladder(cmf, subsets=None, order=1, otxc='tPBE', ot=None, grids_level=None, rdms=None, verbose=None,
                    **kwargs):
    """
    cMF-PDFT spin ladder with the RDMs, not the energies, expanded over sets of clusters (`mbe_rdm12`): E_ot is
    evaluated once per spin state on the assembled density and on-top pair density, which treats the nonlinearity of
    the functional exactly. Adding the J (or energy) increments of separate PDFT calculations would instead rely on
    E_ot being linear in the changes of Pi.

    Parameters
    ----------
    subsets : list
        See `mbe_rdm12`
    rdms : dict
        Output of `mbe_rdm12`, to reuse it for several functionals

    Returns
    -------
    energies : dict
        S -> E(S)
    j : float
        J of H = -2J S_A.S_B (Hartree)
    """
    log = logger.new_logger(cmf, verbose)
    if rdms is None:
        rdms = mbe_rdm12(cmf, subsets, order, verbose, **kwargs)
    if ot is None:
        ot = make_ot(cmf.mol, otxc, grids_level)
    energies = {s: energy(cmf, rdm12=rdm, ot=ot)[0] for s, rdm in rdms.items()}
    s = np.array(sorted(energies))
    e = np.array([energies[x] for x in s])
    j = np.linalg.lstsq(np.vstack((np.ones_like(s), -s*(s+1))).T, e, rcond=None)[0][1]
    log.note('cMF-PDFT (%s, RDM expansion) J = %.3f cm-1', ot.otxc, j*219474.6313705)
    return energies, j
