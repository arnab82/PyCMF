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


def _reference_rdms(cmf, orbs, h2_0, spin_average, max_states, log):
    """
    Spin-free 1-, 2-, 3-RDMs (`fci.rdm.make_dm123` convention) of the current cMF state in the active orbitals `orbs`,
    and the contractions of the 4-RDM with the intra-cluster integrals that NEVPT2 needs (f3ca, f3ac).
    With `spin_average`, the average over the Ms components of every cluster (the Ms ensemble of RO-cMF).
    """
    sizes = [len(c) for c in cmf.clusters]
    norb = len(orbs)
    multiplets = [_multiplet(cmf.ci[i], sizes[i], tuple(cmf.fspace[i])) for i in range(len(sizes))]
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
        members = [[(tuple(f), c) for f, c in zip(cmf.fspace, cmf.ci)]]
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


def nevpt2(cmf, spin_average=True, max_states=64, verbose=None) -> (float, dict):
    """
    SC-NEVPT2 correlation energy of the current cMF state (for `state_average_`, of the selected state)

    Parameters
    ----------
    spin_average : bool
        Use the Ms ensemble of each cluster (RO-cMF) as the reference, instead of the product state of `fspace`
    max_states : int
        Largest number of Ms combinations averaged over

    Returns
    -------
    e_corr : float
    parts : dict
        Energy of each class of perturbers
    """
    log = logger.new_logger(cmf, verbose)
    mf, mol = cmf.mf, cmf.mol
    orbs = np.asarray(sum([list(c) for c in cmf.clusters], []), dtype=int)
    Cact = cmf.Cact[:, orbs]
    ncore, ncas = cmf.Cdoc.shape[1], Cact.shape[1]
    nelecas = (sum(f[0] for f in cmf.fspace), sum(f[1] for f in cmf.fspace))

    # semicanonical doubly occupied and virtual orbitals of the Fock matrix of the cMF density
    hcore = mf.get_hcore()
    vj, vk = mf.get_jk(mol, cmf.get_core_dm() + sum(cmf.get_cluster_dms()), hermi=1)
    fock = hcore + vj - .5*vk
    e_core, u = np.linalg.eigh(cmf.Cdoc.T @ fock @ cmf.Cdoc)
    moc = cmf.Cdoc @ u
    e_vir, u = np.linalg.eigh(cmf.Cvir.T @ fock @ cmf.Cvir)
    mov = cmf.Cvir @ u
    mo = np.hstack((moc, Cact, mov))
    nocc = ncore + ncas

    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.ncore = ncore                    # frozen orbitals are not part of the NEVPT2 core
    mc.mo_coeff = mo
    nev = mrpt.NEVPT(mc)
    nev.verbose = log.verbose
    nev.canonicalized = True
    nev.mo_coeff = mo
    nev.mo_energy = np.concatenate((e_core, np.diag(Cact.T @ fock @ Cact), e_vir))

    # integrals, with the frozen and doubly occupied orbitals in h1eff
    eris = {'ppaa': cmf.ao2mo((mo, mo, Cact, Cact)),
            'papa': cmf.ao2mo((mo, Cact, mo, Cact)),
            'pacv': cmf.ao2mo((mo, Cact, moc, mov))}
    vj, vk = mf.get_jk(mol, cmf.get_core_dm(), hermi=1)
    eris['h1eff'] = mo.T @ (hcore + vj - .5*vk) @ mo

    # active part of H0: the embedded cluster Hamiltonians
    h1 = eris['h1eff'][ncore:nocc, ncore:nocc]
    h2 = eris['ppaa'][ncore:nocc, ncore:nocc]
    offsets = np.cumsum([0] + [len(c) for c in cmf.clusters])
    h1_0 = np.zeros_like(h1)
    h2_0 = np.zeros_like(h2)
    for n in range(len(cmf.clusters)):
        sn = slice(offsets[n], offsets[n+1])
        f = h1[sn, sn].copy()
        for m in range(len(cmf.clusters)):
            if m != n:
                sm = slice(offsets[m], offsets[m+1])
                g = cmf.cluster_1rdm[m]
                f += np.einsum('pqvw,vw->pq', h2[sn, sn, sm, sm], g) - .5*np.einsum('pwvq,vw->pq', h2[sn, sm, sm, sn], g)
        h1_0[sn, sn] = f
        h2_0[sn, sn, sn, sn] = h2[sn, sn, sn, sn]
    eris['h1eff'][ncore:nocc, ncore:nocc] = h1_0
    eris['ppaa'][ncore:nocc, ncore:nocc] = h2_0

    dms, nmembers = _reference_rdms(cmf, orbs, np.ascontiguousarray(h2_0), spin_average, max_states, log)

    parts = {}
    parts['Sr'] = _nevpt2.Sr(nev, None, dms, eris)[1]
    parts['Si'] = _nevpt2.Si(nev, None, dms, eris)[1]
    parts['Sijrs'] = _sijrs(cmf, moc, mov, e_core, e_vir)
    parts['Sijr'] = _nevpt2.Sijr(nev, dms, eris)[1]
    parts['Srsi'] = _nevpt2.Srsi(nev, dms, eris)[1]
    parts['Srs'] = _nevpt2.Srs(nev, dms, eris)[1]
    parts['Sij'] = _nevpt2.Sij(nev, dms, eris)[1]
    parts['Sir'] = _nevpt2.Sir(nev, dms, eris)[1]
    e_corr = sum(parts.values())
    log.info('cMF-NEVPT2: %d frozen, %d core, %d active, %d virtual orbitals, %d reference product states',
             cmf.Cfrz.shape[1], ncore, ncas, mov.shape[1], nmembers)
    for k, v in parts.items():
        log.info('   %-6s %.12f', k, v)
    log.note('cMF-NEVPT2 correlation energy = %.12f', e_corr)
    return e_corr, parts
