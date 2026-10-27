"""
Second-order perturbation theory on top of cMF

The cMF product state is an eigenfunction of H0 = sum of the embedded cluster Hamiltonians (each cluster in the
mean field of the others). Product states that are degenerate in H0 with the reference, the Ms components of the
cluster spin multiplets and, for state-averaged runs, the other charge configurations, form the model space, and a
second-order effective Hamiltonian in this space is diagonalized:

    H_eff[a,b] = <a|H|b> + 1/2 sum_k <a|H|k><k|H|b> (1/(E_a - E_k) + 1/(E_b - E_k))

where k runs over the products of cluster eigenstates outside the model space. Spin coupling and electron hopping
between the clusters are treated exactly in the model space, everything else to second order.

partition = 'en' (Epstein-Nesbet): E_k = <k|H|k>, which includes the interaction between the clusters in the states
    k, e.g. the attraction of the electron and hole of a charge-transfer state. This is needed for kinetic exchange.
    The exchange between clusters in E_k is taken from spin-summed densities, as in the Ms ensemble of RO-cMF, so
    that H_eff stays spin adapted.
partition = 'cmf': E_k = sum of the cluster energies in H0.

The couplings <k|H|a> are obtained from H|a> in the determinant space of the selected clusters (PySCF FCI
sigma-vector) projected on the eigenstates of each cluster, so this is meant for a few clusters at a time, e.g. a
pair; `pair_pt2` sums pair contributions for large systems. Clusters too large to diagonalize fully use an iterative
solution of the first-order equations instead (partition 'cmf' only).
"""
import itertools
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from pyscf import fci, lib
from pyscf.fci import cistring, direct_spin1
from pyscf.lib import logger


class QDPT2Result:
    """Result of `qdpt2`

    Attributes
    ----------
    e_cmf : float
        cMF energy of the reference product state
    heff : np.ndarray
        Second-order effective Hamiltonian in the model space (total energies)
    h1st : np.ndarray
        First-order part, <a|H|b>
    model : list
        For each model function, the (alpha, beta) electrons of each cluster
    energies, vectors : np.ndarray
        Eigenvalues and eigenvectors of heff
    s2 : np.ndarray
        <S^2> of the zeroth-order part of each eigenvector
    e2 : float
        Second-order correction of the reference product state, heff[0,0] - h1st[0,0]
    """
    def __init__(self, heff, h1st, model, s2=None, partition=None):
        self.e_cmf = h1st[0, 0]
        self.heff = heff
        self.h1st = h1st
        self.model = model
        self.energies, self.vectors = scipy.linalg.eigh(heff)
        self.s2 = s2
        self.e2 = heff[0, 0] - h1st[0, 0]
        self.partition = partition

    @property
    def spins(self) -> np.ndarray:
        """Total spin S of each eigenvector, from <S^2> = S(S+1)"""
        return .5*(np.sqrt(1 + 4*np.maximum(self.s2, 0)) - 1)

    def heisenberg_j(self) -> float:
        """
        Exchange coupling J of H = -2J S_A.S_B fitted to the energies, E(S) = E0 - J S(S+1)
        (positive J: ferromagnetic). Each total spin is counted once, with its lowest energy.
        """
        levels = {}
        for e, s in zip(self.energies, np.round(2*self.spins)/2):
            levels[s] = min(e, levels.get(s, np.inf))
        s = np.array(sorted(levels))
        e = np.array([levels[si] for si in s])
        a = np.vstack((np.ones_like(s), -s*(s+1))).T
        return np.linalg.lstsq(a, e, rcond=None)[0][1]


def _spin_ladder(ci, norb, nelec, raise_):
    """S^+ |ci> (raise_) or S^- |ci>, normalized
    """
    na, nb = nelec
    out = 0
    for p in range(norb):
        if raise_:
            if nb == 0 or na == norb:
                return None
            tmp = fci.addons.des_b(ci, norb, (na, nb), p)
            out = out + fci.addons.cre_a(tmp, norb, (na, nb-1), p)
        else:
            if na == 0 or nb == norb:
                return None
            tmp = fci.addons.des_a(ci, norb, (na, nb), p)
            out = out + fci.addons.cre_b(tmp, norb, (na-1, nb), p)
    norm = np.linalg.norm(out)
    return None if norm < 1e-8 else out / norm


def _multiplet(ci, norb, nelec) -> dict:
    """All Ms components of the spin multiplet of `ci`, as {(na, nb): ci}
    """
    s2 = fci.spin_square(ci, norb, nelec)[0]
    s = .5*(np.sqrt(1 + 4*max(s2, 0)) - 1)
    out = {tuple(nelec): ci}
    for raise_ in (True, False):
        c, (na, nb) = ci, tuple(nelec)
        while abs(.5*(na - nb) + (1 if raise_ else -1)) <= s + 1e-6:
            c = _spin_ladder(c, norb, (na, nb), raise_)
            if c is None:
                break
            na, nb = (na+1, nb-1) if raise_ else (na-1, nb+1)
            out[(na, nb)] = c
    return out


def _full_strings(norbs, nstr, offsets):
    """Strings in the space of all clusters for every combination of cluster strings, shape (s_1, ..., s_n)
    """
    out = cistring.make_strings(range(norbs[0]), nstr[0])
    for n, k, off in zip(norbs[1:], nstr[1:], offsets[1:]):
        out = out[..., None] | (cistring.make_strings(range(n), k) << off)
    return out


def _sector_sign(secs) -> int:
    """Sign between (a_1 b_1)(a_2 b_2)... and (a_1 a_2 ...)(b_1 b_2 ...) ordering of the creation operators
    """
    sign = 1
    for i in range(len(secs)):
        for j in range(i+1, len(secs)):
            sign *= (-1)**(secs[i][1]*secs[j][0])
    return sign


def product_ci(cis, norbs, nelecs) -> np.ndarray:
    """CI vector of a product of cluster states, in the determinant space of all their orbitals (in order)

    Parameters
    ----------
    cis : list
        CI vector (na_strings, nb_strings) of each cluster
    norbs : list
        Number of orbitals of each cluster
    nelecs : list
        (alpha, beta) electrons of each cluster
    """
    offsets = np.cumsum([0] + list(norbs))
    norb = int(offsets[-1])
    na, nb = sum(s[0] for s in nelecs), sum(s[1] for s in nelecs)
    addr_a = cistring.strs2addr(norb, na, _full_strings(norbs, [s[0] for s in nelecs], offsets).ravel())
    addr_b = cistring.strs2addr(norb, nb, _full_strings(norbs, [s[1] for s in nelecs], offsets).ravel())
    t = np.asarray(cis[0])
    for c in cis[1:]:
        t = np.multiply.outer(t, np.asarray(c))
    # (a_1, b_1, a_2, b_2, ...) -> (a_1, a_2, ..., b_1, b_2, ...)
    n = len(cis)
    t = t.transpose(list(range(0, 2*n, 2)) + list(range(1, 2*n, 2)))
    out = np.zeros((cistring.num_strings(norb, na), cistring.num_strings(norb, nb)))
    out[np.ix_(addr_a, addr_b)] = _sector_sign(nelecs) * t.reshape(len(addr_a), len(addr_b))
    return out


def _cluster_basis(h1, eri, norb, nelec):
    """Eigenvalues, eigenvectors (columns) and spin-resolved 1RDMs (state, spin, p, q) of a cluster Hamiltonian
    """
    na, nb = cistring.num_strings(norb, nelec[0]), cistring.num_strings(norb, nelec[1])
    dim = na * nb
    if dim == 1:
        e = np.array([direct_spin1.energy(h1, eri, np.ones((1, 1)), norb, nelec)])
        X = np.ones((1, 1))
    else:
        addr, h = direct_spin1.pspace(h1, eri, norb, nelec, np=dim)
        H = np.zeros((dim, dim))
        H[np.ix_(addr, addr)] = h
        e, X = np.linalg.eigh(H)
    return e, X, _rdm1s(X, norb, nelec)


def _rdm1s(X, norb, nelec):
    na, nb = cistring.num_strings(norb, nelec[0]), cistring.num_strings(norb, nelec[1])
    return np.array([direct_spin1.make_rdm1s(X[:, k].reshape(na, nb), norb, nelec) for k in range(X.shape[1])])


def _adapt_basis(e, X, cis, tol=1e-7) -> list:
    """Rotate the eigenvectors within degenerate groups so that the given (eigen)vectors are among them,
    and return their indices
    """
    idx = []
    groups = {}
    for c in cis:
        x = X.T @ c.ravel()
        k = int(np.argmax(abs(x)))
        grp = tuple(np.where(abs(e - e[k]) < tol)[0])
        groups.setdefault(grp, []).append(len(idx))
        idx.append(k)
    for grp, members in groups.items():
        grp = list(grp)
        if len(grp) == 1:
            continue
        sub = X[:, grp]
        vecs = np.array([sub.T @ cis[m].ravel() for m in members]).T
        q, _ = np.linalg.qr(np.hstack((vecs, np.eye(len(grp)))))
        X[:, grp] = sub @ q[:, :len(grp)]
        for n, m in enumerate(members):
            idx[m] = grp[n]
    return idx


def _setup(cmf, clusters, fspaces, eri_act, vcore):
    """Hamiltonian of the selected clusters, H0 and the configurations of the model space
    """
    if eri_act is None:
        eri_act = cmf.get_active_eris()
    h0, h1, h2, orbs = cmf.get_active_hamiltonian(clusters, eri_act, vcore)
    sizes = [len(cmf.clusters[i]) for i in clusters]
    offsets = np.cumsum([0] + sizes)

    # densities of the selected clusters for H0, averaged over the states of a state-averaged run
    rdm1s = [0 for i in clusters]
    for k, w in cmf._states():
        if k is not None:
            cmf.select_state(k)
        for n, i in enumerate(clusters):
            rdm1s[n] = rdm1s[n] + w * cmf.cluster_1rdm[i]
    if fspaces is None:
        if cmf.sa_states is None:
            fspaces = [[cmf.fspace[i] for i in clusters]]
        else:
            fspaces = [[st['fspace'][i] for i in clusters] for st in cmf.sa_states]
    if cmf.sa_states is not None:
        cmf.select_state(0)

    # embedding of each cluster by the other selected clusters, and the cluster Hamiltonians of H0
    vref, h1_0 = [], np.zeros_like(h1)
    h2_0 = np.zeros_like(h2)
    for n in range(len(clusters)):
        sn = slice(offsets[n], offsets[n+1])
        v = np.zeros((sizes[n], sizes[n]))
        for m in range(len(clusters)):
            if m != n:
                sm = slice(offsets[m], offsets[m+1])
                v += np.einsum('pqvw,vw->pq', h2[sn, sn, sm, sm], rdm1s[m])
                v -= .5*np.einsum('pwvq,vw->pq', h2[sn, sm, sm, sn], rdm1s[m])
        vref.append(v)
        h1_0[sn, sn] = h1[sn, sn] + v
        h2_0[sn, sn, sn, sn] = h2[sn, sn, sn, sn]
    return dict(h0=h0, h1=h1, h2=h2, sizes=sizes, offsets=offsets, norb=int(offsets[-1]),
                vref=vref, h1_0=h1_0, h2_0=h2_0, fspaces=[[tuple(s) for s in f] for f in fspaces])


def _model_space(s, solver):
    """Model functions: products of the Ms components of the cluster ground states of each configuration,
    with the total Ms of the first configuration. Returns the sectors and cluster CI vectors of each function.
    """
    sizes, offsets = s['sizes'], s['offsets']
    multiplets = []      # per cluster: list of multiplets {(na, nb): ci}
    for n in range(len(sizes)):
        multiplets.append([])
    for f in s['fspaces']:
        for n, nelec in enumerate(f):
            if any(nelec in m for m in multiplets[n]):
                continue
            sn = slice(offsets[n], offsets[n+1])
            e, ci = solver.kernel(s['h1_0'][sn, sn], s['h2_0'][sn, sn, sn, sn], sizes[n], nelec)
            multiplets[n].append(_multiplet(ci, sizes[n], nelec))
    nelec_tot = (sum(x[0] for x in s['fspaces'][0]), sum(x[1] for x in s['fspaces'][0]))
    model, model_cis = [], []
    for f in s['fspaces']:
        options = [list(next(m for m in multiplets[n] if nelec in m).items()) for n, nelec in enumerate(f)]
        for combo in itertools.product(*options):
            secs = tuple(sec for sec, c in combo)
            if (sum(x[0] for x in secs), sum(x[1] for x in secs)) != nelec_tot or secs in model:
                continue
            model.append(secs)
            model_cis.append([c for sec, c in combo])
    return model, model_cis, nelec_tot


def qdpt2(cmf, clusters=None, fspaces=None, partition='en', level_shift=0.0, max_dim=5000,
          eri_act=None, vcore=None, conv_tol=1e-9, max_cycle=1000, verbose=None) -> QDPT2Result:
    """
    Quasi-degenerate PT2 for the clusters `clusters` (default: all) of a converged `cmf`, with the other
    clusters folded into the mean field (see `CMF.get_active_hamiltonian`).

    Parameters
    ----------
    clusters : list
        Indices of the clusters
    fspaces : list
        Configurations, (alpha, beta) electrons of each selected cluster, whose spin multiplets span the model
        space. Default: the reference, and for `state_average_` all averaged configurations.
    partition : str
        'en' (Epstein-Nesbet) or 'cmf' denominators, see the module documentation
    level_shift : float
        Subtracted from the denominators, against intruder states
    max_dim : int
        Largest determinant space of one cluster in one sector that is diagonalized fully; larger clusters
        use the iterative solver (partition 'cmf')

    Returns
    -------
    QDPT2Result
    """
    log = logger.new_logger(cmf, verbose)
    if clusters is None:
        clusters = list(range(len(cmf.clusters)))
    clusters = list(clusters)
    s = _setup(cmf, clusters, fspaces, eri_act, vcore)
    sizes, offsets, norb, h0 = s['sizes'], s['offsets'], s['norb'], s['h0']

    solver = fci.direct_spin1.FCISolver()
    solver.conv_tol = cmf.ci_conv_tol
    solver.max_cycle = cmf.ci_max_cycle
    model, model_cis, nelec_tot = _model_space(s, solver)
    nmodel = len(model)

    shape = (cistring.num_strings(norb, nelec_tot[0]), cistring.num_strings(norb, nelec_tot[1]))
    h2e = direct_spin1.absorb_h1e(s['h1'], s['h2'], norb, nelec_tot, .5)
    hop = lambda x: direct_spin1.contract_2e(h2e, x.reshape(shape), norb, nelec_tot).ravel()
    P = np.array([product_ci(c, sizes, secs).ravel() for c, secs in zip(model_cis, model)]).T
    HP = np.array([hop(c) for c in P.T]).T
    h1st = P.T @ HP + h0*np.eye(nmodel)

    largest = max(cistring.num_strings(n, k)**2 for n in sizes for k in range(n+1))
    if largest <= max_dim:
        heff = _heff_tps(s, model, model_cis, nelec_tot, HP, h1st, partition, level_shift, log)
    else:
        if partition != 'cmf':
            log.warn('QDPT2: clusters too large to diagonalize, using partition cmf')
            partition = 'cmf'
        heff = _heff_iterative(s, model_cis, model, nelec_tot, P, HP, h1st, level_shift, conv_tol, max_cycle, log)

    res = QDPT2Result(heff, h1st, model, partition=partition)
    res.s2 = np.array([fci.spin_square((P @ v).reshape(shape), norb, nelec_tot)[0] for v in res.vectors.T])
    log.info('QDPT2 (%s) clusters %s: %d orbitals, %d model functions, E(cMF) = %.12f  E2 = %.3e',
             partition, clusters, norb, nmodel, res.e_cmf, res.e2)
    for e, sp in zip(res.energies, res.spins):
        log.info('   E = %.12f  S = %.2f', e, sp)
    return res


def _heff_tps(s, model, model_cis, nelec_tot, HP, h1st, partition, level_shift, log):
    """Second-order H_eff from H|a> projected on products of cluster eigenstates
    """
    sizes, offsets, norb, h0, h2 = s['sizes'], s['offsets'], s['norb'], s['h0'], s['h2']
    ncl = len(sizes)
    shape = (cistring.num_strings(norb, nelec_tot[0]), cistring.num_strings(norb, nelec_tot[1]))
    HPm = [HP[:, a].reshape(shape) for a in range(HP.shape[1])]

    # cluster eigenstates in every sector, with the model states among them
    basis = [{} for n in range(ncl)]
    for n in range(ncl):
        sn = slice(offsets[n], offsets[n+1])
        for na in range(sizes[n]+1):
            for nb in range(sizes[n]+1):
                basis[n][(na, nb)] = list(_cluster_basis(s['h1_0'][sn, sn], s['h2_0'][sn, sn, sn, sn], sizes[n], (na, nb)))
    model_idx = [[None]*ncl for a in model]
    for n in range(ncl):
        for sec in basis[n]:
            members = [a for a, secs in enumerate(model) if secs[n] == sec]
            if not members:
                continue
            e, X, dms = basis[n][sec]
            idx = _adapt_basis(e, X, [model_cis[a][n] for a in members])
            basis[n][sec][2] = _rdm1s(X, sizes[n], sec)
            for a, k in zip(members, idx):
                model_idx[a][n] = k

    # intercluster Coulomb and exchange integrals for the Epstein-Nesbet diagonal
    jk = {}
    for n in range(ncl):
        for m in range(n+1, ncl):
            sn, sm = slice(offsets[n], offsets[n+1]), slice(offsets[m], offsets[m+1])
            jk[(n, m)] = (h2[sn, sn, sm, sm].reshape(sizes[n]**2, sizes[m]**2)
                          - .5*h2[sn, sm, sm, sn].transpose(0, 3, 2, 1).reshape(sizes[n]**2, sizes[m]**2))

    def zeroth_order(secs):
        """Zeroth-order energies of all products of cluster eigenstates in the sectors `secs`"""
        if partition == 'cmf':
            D = basis[0][secs[0]][0]
            for n in range(1, ncl):
                D = np.add.outer(D, basis[n][secs[n]][0])
            return D
        # Epstein-Nesbet: <k|H|k> with the exchange between clusters from spin-summed densities (as in the
        # Ms ensemble of RO-cMF), so that all Ms components of a spin multiplet have the same energy
        dens = [basis[n][secs[n]][2].sum(axis=1).reshape(-1, sizes[n]**2) for n in range(ncl)]
        D = basis[0][secs[0]][0] - dens[0] @ s['vref'][0].ravel() + h0
        for n in range(1, ncl):
            D = np.add.outer(D, basis[n][secs[n]][0] - dens[n] @ s['vref'][n].ravel())
        for n in range(ncl):
            for m in range(n+1, ncl):
                pair = dens[n] @ jk[(n, m)] @ dens[m].T
                shp = [1]*ncl
                shp[n], shp[m] = pair.shape
                D = D + pair.reshape(shp)
        return D

    e_ref = np.array([zeroth_order(secs)[tuple(model_idx[a])] for a, secs in enumerate(model)])

    nmodel = len(model)
    heff = h1st.copy()
    nsmall = 0
    sectors = [list(basis[n]) for n in range(ncl)]
    for secs in itertools.product(*sectors):
        if (sum(x[0] for x in secs), sum(x[1] for x in secs)) != nelec_tot:
            continue
        addr_a = cistring.strs2addr(norb, nelec_tot[0], _full_strings(sizes, [x[0] for x in secs], offsets).ravel())
        addr_b = cistring.strs2addr(norb, nelec_tot[1], _full_strings(sizes, [x[1] for x in secs], offsets).ravel())
        dims_a = [cistring.num_strings(n, x[0]) for n, x in zip(sizes, secs)]
        dims_b = [cistring.num_strings(n, x[1]) for n, x in zip(sizes, secs)]
        perm = [i for n in range(ncl) for i in (n, ncl+n)]
        sign = _sector_sign(secs)
        C = []
        for hv in HPm:
            t = hv[np.ix_(addr_a, addr_b)].reshape(dims_a + dims_b).transpose(perm)
            t = sign * t.reshape([da*db for da, db in zip(dims_a, dims_b)])
            for n in range(ncl):
                t = np.tensordot(t, basis[n][secs[n]][1], axes=([0], [0]))
            C.append(t)
        if not any(np.any(abs(c) > 1e-14) for c in C):
            continue
        D = zeroth_order(secs)

        mask = np.ones(D.shape, bool)
        for a, ms in enumerate(model):
            if ms == secs:
                mask[tuple(model_idx[a])] = False
        C = np.array([c[mask] for c in C])
        D = D[mask]
        W = np.zeros_like(C)
        for a in range(nmodel):
            denom = e_ref[a] - D - level_shift
            small = abs(denom) < 1e-10
            nsmall += np.count_nonzero(small & (abs(C[a]) > 1e-10))
            W[a] = np.where(small, 0, C[a] / np.where(small, 1, denom))
        heff += .5*(W @ C.T + C @ W.T)
    if nsmall:
        log.warn('QDPT2: %d couplings to states degenerate with the model space were skipped', nsmall)
    return heff


def _heff_iterative(s, model_cis, model, nelec_tot, P, HP, h1st, level_shift, conv_tol, max_cycle, log):
    """Second-order H_eff (partition cmf) from the first-order equations, solved in the determinant space
    """
    norb = s['norb']
    shape = (cistring.num_strings(norb, nelec_tot[0]), cistring.num_strings(norb, nelec_tot[1]))
    h2e_0 = direct_spin1.absorb_h1e(s['h1_0'], s['h2_0'], norb, nelec_tot, .5)
    hop0 = lambda x: direct_spin1.contract_2e(h2e_0, x.reshape(shape), norb, nelec_tot).ravel()
    hdiag0 = direct_spin1.make_hdiag(s['h1_0'], s['h2_0'], norb, nelec_tot)
    e0 = np.array([c @ hop0(c) for c in P.T])

    rhs = HP - P @ (P.T @ HP)      # Q H |a>
    psi1 = np.zeros_like(rhs)
    project = lambda x: x - P @ (P.T @ x)
    n = P.shape[0]
    for a in range(P.shape[1]):
        shift = -level_shift - e0[a]
        op = scipy.sparse.linalg.LinearOperator((n, n), lambda x: project(hop0(project(x)) + shift*project(x)))
        d = hdiag0 + shift
        d = np.maximum(abs(d), 1e-2)
        pre = scipy.sparse.linalg.LinearOperator((n, n), lambda x: project(x / d))
        x, info = scipy.sparse.linalg.minres(op, -rhs[:, a], rtol=conv_tol, maxiter=max_cycle, M=pre)
        if info != 0:
            log.warn('QDPT2: first-order equation %d not converged', a)
        psi1[:, a] = project(x)
    v12 = rhs.T @ psi1
    return h1st + .5*(v12 + v12.T)


def cluster_centroids(cmf) -> np.ndarray:
    """Center (Angstrom) of the orbitals of each cluster, <r> averaged over its orbitals
    """
    r = cmf.mol.intor_symmetric('int1e_r')
    out = []
    for ci in cmf.clusters:
        c = cmf.Cact[:, ci]
        out.append(np.einsum('xmn,mt,nt->x', r, c, c) / len(ci))
    return np.array(out) * lib.param.BOHR


def pair_pt2(cmf, pairs=None, cutoff=None, **kwargs) -> (float, dict):
    """
    Pair-additive cMF-PT2: E = E_cMF + sum over cluster pairs of (lowest QDPT2 energy of the pair - E_cMF)

    Each pair is treated with `qdpt2` in the mean field of the other clusters, so the pair contribution includes
    the second-order correlation within the pair and, for open-shell clusters, the stabilization of the lowest spin
    state of the pair. For more than two spin-coupled clusters the pairs are not additive; use the pair
    couplings (`QDPT2Result.heisenberg_j`) to build a spin Hamiltonian instead.

    Parameters
    ----------
    pairs : list
        Pairs of cluster indices, default all pairs
    cutoff : float
        Skip pairs whose cluster centroids are farther apart (Angstrom)
    kwargs
        Passed to `qdpt2`

    Returns
    -------
    e_tot : float
    increments : dict
        (I, J) -> energy lowering of the pair
    """
    log = logger.new_logger(cmf, kwargs.get('verbose'))
    n = len(cmf.clusters)
    if pairs is None:
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    if cutoff is not None:
        r = cluster_centroids(cmf)
        pairs = [(i, j) for i, j in pairs if np.linalg.norm(r[i] - r[j]) <= cutoff]
    eri_act = cmf.get_active_eris()
    vcore = cmf.get_core_veff()
    increments = {}
    for i, j in pairs:
        res = qdpt2(cmf, [i, j], eri_act=eri_act, vcore=vcore, **kwargs)
        increments[(i, j)] = res.energies[0] - res.e_cmf
        log.info('pair (%d, %d)  dE = %.3e', i, j, increments[(i, j)])
    e_tot = cmf.e_tot + sum(increments.values())
    log.note('pair cMF-PT2: E = %.12f  (%d pairs)', e_tot, len(increments))
    return e_tot, increments
