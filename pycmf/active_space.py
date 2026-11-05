"""
Active-space selection with cMF: start from a generous active space split into clusters (including orbitals that
would normally be doubly occupied or virtual), optimize the orbitals with ooCMF, measure how correlated each orbital
is, keep the correlated ones, and fold the rest back into the doubly occupied and virtual spaces before optimizing
the orbitals of the smaller active space again.

    cmf.kernel()                                    # ooCMF in the large active space
    sel = cmf.select_active_space(occ_thresh=0.02)  # natural occupations and entropies of each cluster
    cmf.reduce_active_space_(sel)                   # smaller clusters; dropped orbitals -> doubly occupied / virtual
    cmf.kernel()                                    # ooCMF in the selected active space

Measures (per cluster natural orbital):

* natural occupation n and single-orbital entropy s = -sum_k w_k ln w_k, with w_k the probabilities of the orbital
  being empty, alpha, beta or doubly occupied (as in autoCAS, Stein and Reiher, JCTC 12, 1760 (2016)). From the cMF
  cluster RDMs these contain the correlation within each cluster exactly and none between clusters.
* with `dm1`, e.g. the 1RDM of `pt2_mbe(..., with_rdm=True)`, the natural orbitals and occupations come from the
  cluster blocks of that 1RDM instead, which adds the first-order correlation between clusters (charge transfer,
  superexchange, dispersion-like pair correlation).

An orbital is kept when its occupation is at least `occ_thresh` away from 0 and 2 (so singly occupied orbitals are
always kept), or, with `entropy_thresh`, when its entropy is at least that; otherwise it becomes doubly occupied
(n > 1) or virtual (n < 1).
"""
import numpy as np
from pyscf import fci, lib
from pyscf.lib import logger


def orbital_entropies(cmf) -> list:
    """
    Natural orbitals of each cluster with their occupations and single-orbital entropies, from the cMF cluster
    wavefunctions (the state selected with `select_state` for a state average)

    Returns
    -------
    list of dict, one per cluster, with
        occ : (n,) natural occupations, descending
        entropy : (n,) single-orbital entropies (nat)
        n_alpha, n_beta, n_double : (n,) <n_a>, <n_b> and <n_a n_b> of each natural orbital
        U : (n, n) natural orbitals in the basis of the cluster's active orbitals (columns)
        C : (nao, n) natural orbitals in the AO basis
    """
    if any(c is None for c in cmf.ci):
        cmf.cmf_ci()
    out = []
    for i, ci in enumerate(cmf.clusters):
        norb, nelec = len(ci), tuple(cmf.fspace[i])
        (d1a, d1b), (_, d2ab, _) = fci.direct_spin1.make_rdm12s(cmf.ci[i], norb, nelec)
        occ, U = np.linalg.eigh(d1a + d1b)
        occ, U = occ[::-1], U[:, ::-1]
        na = np.einsum('pi,pq,qi->i', U, d1a, U)
        nb = np.einsum('pi,pq,qi->i', U, d1b, U)
        # <n_ia n_ib> = dm2ab[i,i,i,i] in the natural orbitals (dm2ab[p,q,r,s] = <p_a' r_b' s_b q_a>)
        nd = np.einsum('pi,qi,ri,si,pqrs->i', U, U, U, U, d2ab)
        w = np.clip(np.array([1 - na - nb + nd, na - nd, nb - nd, nd]), 0, 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            s = -np.sum(np.where(w > 1e-16, w * np.log(w), 0), axis=0)
        out.append({'occ': occ, 'entropy': s, 'n_alpha': na, 'n_beta': nb, 'n_double': nd, 'U': U,
                    'C': cmf.Cact[:, ci] @ U})
    return out


def select_active_space(cmf, occ_thresh=0.02, entropy_thresh=None, dm1=None, verbose=None) -> dict:
    """
    Choose the orbitals of each cluster to keep active

    Parameters
    ----------
    occ_thresh : float
        Keep a natural orbital when min(n, 2 - n) >= occ_thresh
    entropy_thresh : float
        Also keep it when its single-orbital entropy is >= entropy_thresh (cMF RDMs only, not with `dm1`)
    dm1 : np.ndarray
        Spin-summed 1RDM of the active orbitals (nact, nact) to take the natural orbitals and occupations of each
        cluster from (its diagonal block), e.g. from `pt2_mbe(with_rdm=True)`; default: the cMF cluster RDMs

    Returns
    -------
    dict with
        clusters, fspace : of the selected active space (empty clusters removed)
        Cact : (nao, nkept) selected active orbitals (natural orbitals of each cluster)
        Cdoc_add, Cvir_add : orbitals that become doubly occupied / virtual
        table : per cluster, list of (occupation, entropy or None, 'act' | 'doc' | 'vir')
        removed : indices of the clusters with no orbital left
    """
    if cmf.sa_states is not None:
        raise NotImplementedError('active-space selection for a state average')
    log = logger.new_logger(cmf, verbose)
    ent = orbital_entropies(cmf)
    clusters, fspace, Cact, Cdoc, Cvir, table, removed = [], [], [], [], [], [], []
    n0 = 0
    for i, ci in enumerate(cmf.clusters):
        if dm1 is None:
            occ, C, s = ent[i]['occ'], ent[i]['C'], ent[i]['entropy']
        else:
            occ, U = np.linalg.eigh(dm1[np.ix_(ci, ci)])
            occ, U = occ[::-1], U[:, ::-1]
            C, s = cmf.Cact[:, ci] @ U, None
        dist = np.minimum(occ, 2 - occ)
        keep = dist >= occ_thresh
        if entropy_thresh is not None and s is not None:
            keep |= s >= entropy_thresh
        doc = ~keep & (occ > 1)
        vir = ~keep & (occ <= 1)
        na, nb = cmf.fspace[i]
        na, nb = na - doc.sum(), nb - doc.sum()
        nk = int(keep.sum())
        if na < 0 or nb < 0 or na > nk or nb > nk:
            raise ValueError('cluster %d: %d alpha and %d beta electrons do not fit in the %d kept orbitals; '
                             'lower occ_thresh' % (i, na, nb, nk))
        table.append([(occ[k], None if s is None else s[k], 'act' if keep[k] else ('doc' if doc[k] else 'vir'))
                      for k in range(len(ci))])
        Cdoc.append(C[:, doc])
        Cvir.append(C[:, vir])
        if nk == 0:
            if na or nb:
                raise ValueError('cluster %d has electrons left but no orbital' % i)
            removed.append(i)
            continue
        Cact.append(C[:, keep])
        clusters.append(list(range(n0, n0 + nk)))
        fspace.append((int(na), int(nb)))
        n0 += nk
    sel = {'clusters': clusters, 'fspace': fspace, 'Cact': np.hstack(Cact), 'Cdoc_add': np.hstack(Cdoc),
           'Cvir_add': np.hstack(Cvir), 'table': table, 'removed': removed}
    if log.verbose >= logger.NOTE:
        print_selection(cmf, sel, log)
    return sel


def print_selection(cmf, sel, log=None):
    """Table of the natural orbitals of each cluster: occupation, entropy and where each one goes"""
    if log is None:
        log = logger.new_logger(cmf, logger.NOTE)
    log.note('active-space selection: cluster, natural occupations (entropy) -> act | doc | vir')
    for i, rows in enumerate(sel['table']):
        log.note('cluster %d %s:', i, tuple(cmf.fspace[i]))
        for occ, s, where in rows:
            log.note('    %8.5f  %s  %s', occ, '        ' if s is None else '%8.5f' % s, where)
    nact = sel['Cact'].shape[1]
    log.note('selected: (%de, %do) in %d clusters %s (from (%de, %do)); %d orbitals -> doubly occupied, %d -> virtual',
             sum(map(sum, sel['fspace'])), nact, len(sel['clusters']), [len(c) for c in sel['clusters']],
             sum(map(sum, cmf.fspace)), cmf.Cact.shape[1], sel['Cdoc_add'].shape[1], sel['Cvir_add'].shape[1])


def reduce_active_space_(cmf, sel) -> None:
    """
    Make the selection of `select_active_space` the active space of `cmf`: the dropped orbitals join the doubly
    occupied and virtual spaces (so ooCMF optimizes them with the others), and the cluster 1RDMs start from the
    natural occupations of the kept orbitals. Run `cmf.kernel()` afterwards to optimize the orbitals again
    """
    occ = [np.array([o for o, _, w in rows if w == 'act']) for i, rows in enumerate(sel['table'])
           if i not in sel['removed']]
    cmf.set_Cdoc(np.hstack((cmf.Cdoc, sel['Cdoc_add'])))
    cmf.set_Cvir(np.hstack((sel['Cvir_add'], cmf.Cvir)))
    cmf.set_Cact(sel['Cact'], [list(c) for c in sel['clusters']], [tuple(f) for f in sel['fspace']])
    cmf.cluster_1rdm = [np.diag(o) for o in occ]
    cmf.converged = False
