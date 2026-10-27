"""
Dynamic correlation on top of cMF: strongly contracted NEVPT2

The zeroth-order Hamiltonian is Dyall's, with the active part replaced by the sum of the embedded cluster
Hamiltonians (each cluster in the mean field of the core and the other clusters). The cMF state is an eigenfunction
of this H0, as NEVPT2 requires, and the correlation between the clusters themselves, excitations within the active
orbitals, is left to `qdpt2`, which uses the same cluster Hamiltonians. The perturbers are those of SC-NEVPT2
(excitations involving the doubly occupied and virtual orbitals) and the energy expressions are those of
`pyscf.mrpt.nevpt2`, with the active-space RDMs of the cMF product state. These are built in the determinant space of
the active orbitals, so the active space has to be moderate (about 14 orbitals); the doubly occupied and virtual
spaces can be large: frozen orbitals (`set_Cfrz`) are not correlated, and with density fitting the core-core/virtual-
virtual class is evaluated from the 3-index integrals without storing (ia|jb).
"""
import itertools
import numpy as np
from pyscf import fci, lib, mcscf, mrpt
from pyscf.fci import cistring
from pyscf.lib import logger
from pyscf.mrpt import nevpt2 as _nevpt2

from .pt2 import _multiplet, product_ci


def _reference_rdms(cmf, clusters, h2_0, spin_average, max_states, log):
    """
    Spin-free 1-, 2-, 3-RDMs (`fci.rdm.make_dm123` convention) of the current cMF state of the clusters `clusters`,
    in their orbitals (cluster after cluster), and the contractions of the 4-RDM with the intra-cluster integrals
    that NEVPT2 needs (f3ca, f3ac). With `spin_average`, the average over the Ms components of every cluster
    (the Ms ensemble of RO-cMF).
    """
    sizes = [len(cmf.clusters[i]) for i in clusters]
    norb = sum(sizes)
    multiplets = [_multiplet(cmf.ci[i], len(cmf.clusters[i]), tuple(cmf.fspace[i])) for i in clusters]
    nstates = np.prod([len(m) for m in multiplets])
    if spin_average and nstates > max_states:
        log.warn('NEVPT2: %d Ms combinations; using the product state of fspace instead of the Ms ensemble', nstates)
        spin_average = False
    if spin_average:
        # flipping all Ms leaves the spin-free RDMs unchanged, so keep one combination of each pair
        members, weights = [], []
        for combo in itertools.product(*[list(m.items()) for m in multiplets]):
            ms = tuple(a - b for (a, b), c in combo)
            flipped = tuple(-m for m in ms)
            if ms > flipped:
                members.append(combo)
                weights.append(2.)
            elif ms == flipped:
                members.append(combo)
                weights.append(1.)
        weights = np.array(weights) / nstates
    else:
        members = [[(tuple(cmf.fspace[i]), cmf.ci[i]) for i in clusters]]
        weights = np.ones(1)

    dms = {'1': 0, '2': 0, '3': 0, 'f3ca': 0, 'f3ac': 0}
    for w, combo in zip(weights, members):
        secs = [sec for sec, c in combo]
        nelec = (sum(s[0] for s in secs), sum(s[1] for s in secs))
        ci = product_ci([c for sec, c in combo], sizes, secs)
        dm1, dm2, dm3 = fci.rdm.make_dm123('FCI3pdm_kern_sf', ci, ci, norb, nelec)
        link = (cistring.gen_linkstr_index(range(norb), nelec[0]), cistring.gen_linkstr_index(range(norb), nelec[1]))
        dms['1'] = dms['1'] + w*dm1
        dms['2'] = dms['2'] + w*dm2
        dms['3'] = dms['3'] + w*dm3
        dms['f3ca'] = dms['f3ca'] + w*_nevpt2._contract4pdm('NEVPTkern_cedf_aedf', h2_0, ci, norb, nelec, link)
        dms['f3ac'] = dms['f3ac'] + w*_nevpt2._contract4pdm('NEVPTkern_aedf_ecdf', h2_0, ci, norb, nelec, link)
    return dms, len(members)


def _sijrs(cmf, moc, mov, e_core, e_vir):
    """Core-core -> virtual-virtual class, sum_ijab (ia|jb) [2 (ia|jb) - (ib|ja)] / (e_i + e_j - e_a - e_b)
    """
    ncore, nvir = moc.shape[1], mov.shape[1]
    if ncore == 0 or nvir == 0:
        return 0.0
    if hasattr(getattr(cmf.mf, 'with_df', None), 'loop'):
        lov = []
        for eri1 in cmf.mf.with_df.loop():
            lov.append(lib.einsum('Lpq,pi,qa->Lia', lib.unpack_tril(eri1), moc, mov))
        lov = np.concatenate(lov)
        ovov = lambda i: lib.einsum('La,Ljb->jab', lov[:, i], lov)
    else:
        g = cmf.ao2mo((moc, mov, moc, mov))
        ovov = lambda i: g[i].transpose(1, 0, 2)
    e = 0.0
    for i in range(ncore):
        gi = ovov(i)                                  # (ia|jb) as [j, a, b]
        denom = e_core[i] + e_core[:, None, None] - e_vir[None, :, None] - e_vir[None, None, :]
        e += np.einsum('jab,jab', gi / denom, 2*gi - gi.transpose(0, 2, 1))
    return e


def _veff(cmf, clusters):
    """J - K/2 (AO) of the frozen and doubly occupied orbitals and of the clusters `clusters`, built from
    orbitals and occupations so that density-fitted K is cheap
    """
    orbs = [np.hstack((cmf.Cfrz, cmf.Cdoc))]
    occs = [np.full(orbs[0].shape[1], 2.0)]
    for i in clusters:
        c, n = cmf._cluster_natural_orbitals(i, cmf.cluster_1rdm[i])
        orbs.append(c)
        occs.append(n)
    return cmf._get_veff([np.hstack(orbs)], [np.concatenate(occs)])[0]


# Sijr, Srsi and Sir of pyscf.mrpt.nevpt2 (Apache License 2.0), with the contractions of three tensors done in
# an optimized order; the expressions are unchanged


def _sijr(mc, dms, eris):
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    dm1, dm2 = dms['1'], dms['2']
    h1e = eris['h1eff'][ncore:nocc, ncore:nocc]
    h2e = eris['ppaa'][ncore:nocc, ncore:nocc].transpose(0, 2, 1, 3)
    h2e_v = eris['pacv'][:ncore].transpose(3, 1, 2, 0)
    hdm1 = _nevpt2.make_hdm1(dm1)
    a3 = _nevpt2.make_a3(h1e, h2e, dm1, dm2, hdm1)
    ci_diag = np.diag_indices(ncore)
    ci_triu = np.triu_indices(ncore)

    def pair_sum(d):
        x = np.einsum('raji,pa->rpji', h2e_v, d, optimize=True)
        out = 2.0*np.einsum('rpji,rpji->rji', h2e_v, x) - np.einsum('rpji,rpij->rji', h2e_v, x)
        out += out.transpose(0, 2, 1)
        out[:, ci_diag[0], ci_diag[1]] *= 0.5
        return out[:, ci_triu[0], ci_triu[1]]
    diff = mc.mo_energy[nocc:, None, None] - mc.mo_energy[None, :ncore, None] - mc.mo_energy[None, None, :ncore]
    return _nevpt2._norm_to_energy(pair_sum(hdm1), pair_sum(a3), diff[:, ci_triu[0], ci_triu[1]])[1]


def _srsi(mc, dms, eris):
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    nvirt = mc.mo_coeff.shape[1] - nocc
    dm1, dm2 = dms['1'], dms['2']
    h1e = eris['h1eff'][ncore:nocc, ncore:nocc]
    h2e = eris['ppaa'][ncore:nocc, ncore:nocc].transpose(0, 2, 1, 3)
    h2e_v = eris['pacv'][nocc:].transpose(3, 0, 2, 1)
    k27 = _nevpt2.make_k27(h1e, h2e, dm1, dm2)
    vi_diag = np.diag_indices(nvirt)
    vi_triu = np.triu_indices(nvirt)

    def pair_sum(d):
        x = np.einsum('rsia,pa->rsip', h2e_v, d, optimize=True)
        out = 2.0*np.einsum('rsip,rsip->rsi', h2e_v, x) - np.einsum('rsip,srip->rsi', h2e_v, x)
        out += out.transpose(1, 0, 2)
        out[vi_diag] *= 0.5
        return out[vi_triu]
    diff = mc.mo_energy[nocc:, None, None] + mc.mo_energy[None, nocc:, None] - mc.mo_energy[None, None, :ncore]
    return _nevpt2._norm_to_energy(pair_sum(dm1), pair_sum(k27), diff[vi_triu])[1]


def _sir(mc, dms, eris):
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    dm1, dm2, dm3 = dms['1'], dms['2'], dms['3']
    h1e = eris['h1eff'][ncore:nocc, ncore:nocc]
    h2e = eris['ppaa'][ncore:nocc, ncore:nocc].transpose(0, 2, 1, 3)
    h2e_v1 = eris['ppaa'][nocc:, :ncore].transpose(0, 2, 1, 3)
    h2e_v2 = eris['papa'][nocc:, :, :ncore].transpose(0, 3, 1, 2)
    h1e_v = eris['h1eff'][nocc:, :ncore]
    e = lambda sub, *ops: np.einsum(sub, *ops, optimize=True)

    norm = e('rpiq,raib,qpab->ir', h2e_v1, h2e_v1, dm2)*2.0 \
         - e('rpiq,rabi,qpab->ir', h2e_v1, h2e_v2, dm2) \
         - e('rpqi,raib,qpab->ir', h2e_v2, h2e_v1, dm2) \
         + e('raqi,rabi,qb->ir', h2e_v2, h2e_v2, dm1)*2.0 \
         - e('rpqi,rabi,qbap->ir', h2e_v2, h2e_v2, dm2) \
         + e('rpqi,raai,qp->ir', h2e_v2, h2e_v2, dm1) \
         + e('rpiq,ri,qp->ir', h2e_v1, h1e_v, dm1)*4.0 \
         - e('rpqi,ri,qp->ir', h2e_v2, h1e_v, dm1)*2.0 \
         + e('ri,ri->ir', h1e_v, h1e_v)*2.0
    a12 = _nevpt2.make_a12(h1e, h2e, dm1, dm2, dm3)
    a13 = _nevpt2.make_a13(h1e, h2e, dm1, dm2, dm3)
    h = e('rpiq,raib,pqab->ir', h2e_v1, h2e_v1, a12)*2.0 \
      - e('rpiq,rabi,pqab->ir', h2e_v1, h2e_v2, a12) \
      - e('rpqi,raib,pqab->ir', h2e_v2, h2e_v1, a12) \
      + e('rpqi,rabi,pqab->ir', h2e_v2, h2e_v2, a13)
    diff = mc.mo_energy[:ncore, None] - mc.mo_energy[None, nocc:]
    return _nevpt2._norm_to_energy(norm, h, -diff)[1]


def _semicanonical(cmf):
    """Doubly occupied and virtual orbitals that diagonalize the Fock matrix of the cMF density, and their energies
    """
    mf = cmf.mf
    fock = mf.get_hcore() + _veff(cmf, range(len(cmf.clusters)))
    e_core, u = np.linalg.eigh(cmf.Cdoc.T @ fock @ cmf.Cdoc)
    moc = cmf.Cdoc @ u
    e_vir, u = np.linalg.eigh(cmf.Cvir.T @ fock @ cmf.Cvir)
    mov = cmf.Cvir @ u
    return fock, moc, e_core, mov, e_vir


def nevpt2(cmf, spin_average=True, max_states=64, clusters=None, with_sijrs=True, verbose=None,
           _semi=None) -> (float, dict):
    """
    SC-NEVPT2 correlation energy of the current cMF state (for `state_average_`, of the selected state)

    Parameters
    ----------
    spin_average : bool
        Use the Ms ensemble of each cluster (RO-cMF) as the reference, instead of the product state of `fspace`
    max_states : int
        Largest number of Ms combinations averaged over
    clusters : list
        Correlate only these clusters (default: all); the others stay in their cMF states as frozen spectators,
        their mean field included, see `nevpt2_mbe`
    with_sijrs : bool
        Include the core-core/virtual-virtual class, which does not depend on the active orbitals

    Returns
    -------
    e_corr : float
    parts : dict
        Energy of each class of perturbers
    """
    log = logger.new_logger(cmf, verbose)
    mf, mol = cmf.mf, cmf.mol
    if clusters is None:
        clusters = list(range(len(cmf.clusters)))
    clusters = list(clusters)
    spectators = [j for j in range(len(cmf.clusters)) if j not in clusters]
    orbs = np.asarray(sum([list(cmf.clusters[i]) for i in clusters], []), dtype=int)
    Cact = cmf.Cact[:, orbs]
    ncore, ncas = cmf.Cdoc.shape[1], Cact.shape[1]
    nelecas = (sum(cmf.fspace[i][0] for i in clusters), sum(cmf.fspace[i][1] for i in clusters))

    fock, moc, e_core, mov, e_vir = _semicanonical(cmf) if _semi is None else _semi
    mo = np.hstack((moc, Cact, mov))
    nocc = ncore + ncas

    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.ncore = ncore                    # frozen orbitals and spectator clusters are not part of the NEVPT2 core
    mc.mo_coeff = mo
    nev = mrpt.NEVPT(mc)
    nev.verbose = log.verbose
    nev.canonicalized = True
    nev.mo_coeff = mo
    nev.mo_energy = np.concatenate((e_core, np.diag(Cact.T @ fock @ Cact), e_vir))

    # integrals, with the frozen and doubly occupied orbitals and the spectator clusters in h1eff
    eris = {'ppaa': cmf.ao2mo((mo, mo, Cact, Cact)),
            'papa': cmf.ao2mo((mo, Cact, mo, Cact)),
            'pacv': cmf.ao2mo((mo, Cact, moc, mov))}
    eris['h1eff'] = mo.T @ (mf.get_hcore() + _veff(cmf, spectators)) @ mo

    # active part of H0: the embedded cluster Hamiltonians
    h1 = eris['h1eff'][ncore:nocc, ncore:nocc]
    h2 = eris['ppaa'][ncore:nocc, ncore:nocc]
    offsets = np.cumsum([0] + [len(cmf.clusters[i]) for i in clusters])
    h1_0 = np.zeros_like(h1)
    h2_0 = np.zeros_like(h2)
    for n in range(len(clusters)):
        sn = slice(offsets[n], offsets[n+1])
        f = h1[sn, sn].copy()
        for m in range(len(clusters)):
            if m != n:
                sm = slice(offsets[m], offsets[m+1])
                g = cmf.cluster_1rdm[clusters[m]]
                f += np.einsum('pqvw,vw->pq', h2[sn, sn, sm, sm], g) - .5*np.einsum('pwvq,vw->pq', h2[sn, sm, sm, sn], g)
        h1_0[sn, sn] = f
        h2_0[sn, sn, sn, sn] = h2[sn, sn, sn, sn]
    eris['h1eff'][ncore:nocc, ncore:nocc] = h1_0
    eris['ppaa'][ncore:nocc, ncore:nocc] = h2_0

    dms, nmembers = _reference_rdms(cmf, clusters, np.ascontiguousarray(h2_0), spin_average, max_states, log)

    parts = {}
    parts['Sr'] = _nevpt2.Sr(nev, None, dms, eris)[1]
    parts['Si'] = _nevpt2.Si(nev, None, dms, eris)[1]
    if with_sijrs:
        parts['Sijrs'] = _sijrs(cmf, moc, mov, e_core, e_vir)
    parts['Sijr'] = _sijr(nev, dms, eris)
    parts['Srsi'] = _srsi(nev, dms, eris)
    parts['Srs'] = _nevpt2.Srs(nev, dms, eris)[1]
    parts['Sij'] = _nevpt2.Sij(nev, dms, eris)[1]
    parts['Sir'] = _sir(nev, dms, eris)
    e_corr = sum(parts.values())
    log.info('cMF-NEVPT2 for clusters %s: %d frozen, %d core, %d active, %d virtual orbitals, '
             '%d reference product states', clusters, cmf.Cfrz.shape[1], ncore, ncas, mov.shape[1], nmembers)
    for k, v in parts.items():
        log.info('   %-6s %.12f', k, v)
    log.info('cMF-NEVPT2 correlation energy = %.12f', e_corr)
    return e_corr, parts


def nevpt2_mbe(cmf, order=2, cutoff=None, spin_average=True, max_states=64, verbose=None) -> (float, dict):
    """
    cMF-NEVPT2 from a many-body expansion over the clusters, for active spaces too large for `nevpt2`

    E = E(Sijrs) + sum_I e_I + sum_{I<J} (e_IJ - e_I - e_J)

    where e_S is the NEVPT2 energy without the core-core/virtual-virtual class when only the clusters S are
    correlated and the others are frozen spectators (`nevpt2(clusters=S)`). With all orders this is cMF-NEVPT2;
    each term only needs the active space of one or two clusters. Use orbital-optimized cMF (`kernel`): for
    polyenes with three and four clusters the pair truncation is then within 0.1 mEh of cMF-NEVPT2, while with
    non-optimized orbitals the larger couplings between the active and the other orbitals make it less additive.

    Parameters
    ----------
    order : int
        1 (clusters) or 2 (clusters and pairs)
    cutoff : float
        Skip pairs whose cluster centroids are farther apart (Angstrom)

    Returns
    -------
    e_corr : float
    increments : dict
        'sijrs', and (I,) or (I, J) -> increment
    """
    log = logger.new_logger(cmf, verbose)
    n = len(cmf.clusters)
    semi = _semicanonical(cmf)
    fock, moc, e_core, mov, e_vir = semi
    kw = dict(spin_average=spin_average, max_states=max_states, with_sijrs=False, verbose=verbose, _semi=semi)
    increments = {'sijrs': _sijrs(cmf, moc, mov, e_core, e_vir)}
    e1 = {}
    for i in range(n):
        e1[i] = nevpt2(cmf, clusters=[i], **kw)[0]
        increments[(i,)] = e1[i]
    if order >= 2:
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        if cutoff is not None:
            from .pt2 import cluster_centroids
            r = cluster_centroids(cmf)
            pairs = [(i, j) for i, j in pairs if np.linalg.norm(r[i] - r[j]) <= cutoff]
        for i, j in pairs:
            increments[(i, j)] = nevpt2(cmf, clusters=[i, j], **kw)[0] - e1[i] - e1[j]
    e_corr = sum(increments.values())
    log.note('cMF-NEVPT2 (many-body expansion, order %d, %d terms): %.12f', order, len(increments), e_corr)
    return e_corr, increments
