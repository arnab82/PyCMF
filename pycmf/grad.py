"""
Analytic nuclear gradients of orbital-optimized cMF

The cMF energy is variational in all its parameters: each cluster state is an eigenvector of its embedded
Hamiltonian, i.e. stationary for the energy, and all non-redundant orbital rotations are optimized by `CMF.kernel`.
So, as for CASSCF, the gradient needs no response equations:

    dE/dR = E_nuc^R + tr(D h^R) + 1/2 tr(D G^R[D]) - 1/2 sum_I tr(D_I G^R[D_I])
            + 1/2 sum_I (tu|vw)^R G^I_tuvw - tr(S^R W)

where D = D_core + sum_I D_I is the total density, D_I that of cluster I, G[D] = J[D] - K[D]/2, G^I the 2RDM of
cluster I, (tu|vw)^R the derivative AO integrals in fixed orbitals, and W = C (F + F^T)/2 C^T the energy-weighted
density from the generalized Fock matrix F (`CMF.get_gen_fock`). The mean-field terms are J/K derivative builds and
the intra-cluster term uses each cluster's own 2RDM, so the cost does not depend on the coupling between clusters.
With density fitting the derivatives of the fitted integrals are used, including the response of the auxiliary
basis, consistent with the density-fitted energy.

For `state_average_` this is the gradient of the averaged energy, the one that is optimized. Frozen orbitals
(`set_Cfrz`) are not optimized, so their response would be needed; they are not supported.
"""
import numpy as np
from pyscf import ao2mo, gto, lib, scf
from pyscf.lib import logger


def _rhf_grad(cmf):
    """(DF-)RHF gradient object for the molecule of `cmf`, for the one-electron and J/K derivative integrals
    """
    mf = scf.RHF(cmf.mol)
    with_df = getattr(cmf.mf, 'with_df', None)
    if with_df is not None:
        mf = mf.density_fit()
        mf.with_df = with_df
    g = mf.nuc_grad_method()
    g.verbose = 0
    return g


def _state_quantities(cmf):
    """For each state (weight, total AO density, AO density of each cluster) and the averaged intra-cluster 2RDMs
    """
    out = []
    gamma = [0 for c in cmf.clusters]
    dm_core = cmf.get_core_dm()
    for k, w in cmf._states():
        if k is not None:
            cmf.select_state(k)
        dms = cmf.get_cluster_dms()
        out.append((w, dm_core + sum(dms), dms))
        gamma = [g + w*d for g, d in zip(gamma, cmf.cluster_2rdm)]
    if cmf.sa_states is not None:
        cmf.select_state(0)
    return out, gamma


def _intra_cluster_exact(mol, cmf, gamma, atmlst, max_memory):
    """1/2 sum_I (tu|vw)^R G^I_tuvw with the AO derivative integrals (no density fitting)
    """
    nao = mol.nao
    nao_pair = nao*(nao+1)//2
    diag_idx = np.arange(nao)
    diag_idx = diag_idx*(diag_idx+1)//2 + diag_idx
    bufs = []
    for ci, d2 in zip(cmf.clusters, gamma):
        n = len(ci)
        c = cmf.Cact[:, ci]
        d2 = d2 + d2.transpose(0, 1, 3, 2)
        buf = ao2mo._ao2mo.nr_e2(d2.reshape(n*n, n*n), c.T, (0, nao, 0, nao)).reshape(n*n, nao, nao)
        buf = lib.pack_tril(buf)
        buf[:, diag_idx] *= .5
        bufs.append((c, buf.reshape(n, n, nao_pair)))

    aoslices = mol.aoslice_by_atom()
    blksize = int(max_memory*.9e6/8 / (4*(aoslices[:, 3] - aoslices[:, 2]).max()*nao_pair))
    blksize = min(nao, max(2, blksize))
    de = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        q1 = 0
        for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, blksize):
            q0, q1 = q1, q1 + nf
            dm2_ao = 0
            for c, buf in bufs:
                dm2_ao = dm2_ao + lib.einsum('ijw,pi,qj->pqw', buf, c[p0:p1], c[q0:q1])
            shls_slice = (shl0, shl1, b0, b1, 0, mol.nbas, 0, mol.nbas)
            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl', shls_slice=shls_slice).reshape(3, p1-p0, nf, nao_pair)
            de[k] -= np.einsum('xijw,ijw->x', eri1, dm2_ao) * 2
    return de


def _intra_cluster_df(mf_grad, cmf, gamma, atmlst):
    """1/2 sum_I (tu|vw)^R G^I_tuvw with density-fitted integrals, including the auxiliary basis response
    """
    from types import SimpleNamespace
    from pyscf.df.grad import casdm2_util
    nact = cmf.Cact.shape[1]
    view = SimpleNamespace(mol=cmf.mol, max_memory=mf_grad.max_memory, with_df=mf_grad.base.with_df,
                           ncore=0, ncas=nact, nelecas=(0, 0), ci=None, auxbasis_response=mf_grad.auxbasis_response)
    if view.with_df.auxmol is None:
        view.with_df.build()
    # (P|Q) d_Qtu = (P|vw) G_tuvw, solved cluster by cluster: d is block diagonal over the clusters
    naux = view.with_df.auxmol.nao
    dfdm2 = np.zeros((naux, nact, nact))
    for ci, d2 in zip(cmf.clusters, gamma):
        view.ncas = len(ci)
        d = casdm2_util.solve_df_rdm2(view, mo_cas=cmf.Cact[:, ci], casdm2=d2)[0]
        dfdm2[:, np.asarray(ci)[:, None], ci] = d
    view.ncas = nact
    de = casdm2_util.grad_elec_dferi(view, mo_cas=cmf.Cact, dfcasdm2=[dfdm2], atmlst=atmlst,
                                     max_memory=mf_grad.max_memory)[0]
    if mf_grad.auxbasis_response:
        de += casdm2_util.grad_elec_auxresponse_dferi(view, mo_cas=cmf.Cact, dfcasdm2=[dfdm2], atmlst=atmlst,
                                                      max_memory=mf_grad.max_memory)[0]
    return de


def _shell_prange(mol, start, stop, blksize):
    nao = 0
    ib0 = start
    for ib in range(start, stop):
        now = mol.ao_loc[ib+1] - mol.ao_loc[ib]
        nao += now
        if nao > blksize and nao > now:
            yield (ib0, ib, nao-now)
            ib0 = ib
            nao = now
    yield (ib0, stop, nao)


def grad_elec(cmf, atmlst=None, verbose=None) -> np.ndarray:
    """Electronic part of the nuclear gradient, (natm, 3), see the module documentation
    """
    log = logger.new_logger(cmf, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    if cmf.Cfrz.shape[1] > 0:
        raise NotImplementedError('nuclear gradients need all orbitals optimized: frozen orbitals (set_Cfrz) '
                                  'are not supported')
    if cmf.cluster_1rdm is None:
        raise RuntimeError('run CMF.kernel first')
    g = np.linalg.norm(cmf.orbital_gradient())
    if g > 1e-4:
        log.warn('cMF orbital gradient %.1e: the nuclear gradient assumes optimized orbitals (CMF.kernel)', g)
    mol = cmf.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = list(atmlst)
    mf_grad = _rhf_grad(cmf)
    df = hasattr(mf_grad.base, 'with_df')

    states, gamma = _state_quantities(cmf)
    dm1 = sum(w*d for w, d, dms in states)

    # energy-weighted density from the (state-averaged) generalized Fock matrix
    C = cmf.get_mo_coeffs()
    dEdC = cmf._average_deriv(cmf.get_core_veff(), cmf.get_cluster_eris_ao())[0]
    F = .5 * C.T @ dEdC
    dme0 = C @ (.5*(F + F.T)) @ C.T

    # J/K derivatives: total density of each state, and the density of each cluster in each state
    vj, vk = mf_grad.get_jk(mol, np.array([d for w, d, dms in states]))
    vhf_tot = vj - .5*vk
    aux = None
    if df and mf_grad.auxbasis_response:
        a = vj.aux - .5*vk.aux
        aux = sum(w*a[n, n] for n, (w, d, dms) in enumerate(states))
    dms_cl = np.array([d for w, dd, dms in states for d in dms])
    w_cl = np.array([w for w, dd, dms in states for d in dms])
    if len(dms_cl):
        vj, vk = mf_grad.get_jk(mol, dms_cl)
        vhf_cl = vj - .5*vk
        if aux is not None:
            a = vj.aux - .5*vk.aux
            aux -= sum(w*a[n, n] for n, w in enumerate(w_cl))
    t0 = log.timer('cMF gradient: J/K derivatives', *t0)

    if df:
        de = _intra_cluster_df(mf_grad, cmf, gamma, atmlst)
    else:
        de = _intra_cluster_exact(mol, cmf, gamma, atmlst, mf_grad.max_memory - lib.current_memory()[0])
    t0 = log.timer('cMF gradient: intra-cluster 2RDM term', *t0)

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    aoslices = mol.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        de[k] += np.einsum('xij,ij->x', hcore_deriv(ia), dm1)
        de[k] -= np.einsum('xij,ij->x', s1[:, p0:p1], dme0[p0:p1]) * 2
        for n, (w, d, dms) in enumerate(states):
            de[k] += w * np.einsum('xij,ij->x', vhf_tot[n][:, p0:p1], d[p0:p1]) * 2
        for n, w in enumerate(w_cl):
            de[k] -= w * np.einsum('xij,ij->x', vhf_cl[n][:, p0:p1], dms_cl[n][p0:p1]) * 2
        if aux is not None:
            de[k] += aux[ia]
    log.timer('cMF gradient', *t0)
    return de


def kernel(cmf, atmlst=None, verbose=None) -> np.ndarray:
    """Nuclear gradient of the cMF energy, dE/dR (natm, 3) in Hartree/Bohr
    """
    log = logger.new_logger(cmf, verbose)
    if atmlst is None:
        atmlst = range(cmf.mol.natm)
    atmlst = list(atmlst)
    de = grad_elec(cmf, atmlst, verbose) + _rhf_grad(cmf).grad_nuc(cmf.mol, atmlst)
    if log.verbose >= logger.NOTE:
        log.note('--------------- cMF gradients ---------------')
        log.note('         x                y                z')
        for k, ia in enumerate(atmlst):
            log.note('%d %s  %15.10f  %15.10f  %15.10f', ia, cmf.mol.atom_symbol(ia), *de[k])
        log.note('----------------------------------------------')
    return de


def _carry_orbitals(C_old, S):
    """Orbitals of the previous geometry for the new one: the AOs move with the nuclei, so the coefficients are kept
    and orthonormalized symmetrically in the new overlap metric, so that each orbital stays in its cluster
    """
    e, u = np.linalg.eigh(C_old.T @ S @ C_old)
    return C_old @ (u * e**-.5) @ u.T


_SETTINGS = ('conv_tol', 'conv_tol_grad', 'max_cycle', 'max_stepsize', 'lbfgs_memory', 'zero_intra_rots',
             'conv_tol_rdm1', 'max_cycle_rdm1', 'sequential', 'ci_conv_tol', 'ci_max_cycle', 'verbose', 'stdout')


class Gradients(lib.StreamObject):
    """Nuclear gradients of a `CMF` object, `CMF.nuc_grad_method()`; `as_scanner` for `pyscf.geomopt`
    """
    def __init__(self, cmf):
        self.base = cmf
        self.mol = cmf.mol
        self.verbose = cmf.verbose
        self.stdout = cmf.stdout
        self.atmlst = None
        self.de = None

    def kernel(self, atmlst=None):
        if atmlst is None:
            atmlst = self.atmlst
        self.de = kernel(self.base, atmlst, self.verbose)
        return self.de

    def as_scanner(self):
        return GradScanner(self.base)


class GradScanner(lib.GradScanner):
    """
    Energy and nuclear gradient of ooCMF as a function of the geometry, `scanner(mol) -> (e_tot, de)`, e.g. for
    `pyscf.geomopt.geometric_solver.optimize(cmf)` or `berny_solver.optimize(cmf)`.

    At each geometry the SCF is rerun (from the previous density), the clusters are rebuilt from the same AO labels
    and occupations (`CMF.set_clusters_from_ao`, required), with the same state averaging and settings, and the
    orbitals are optimized before the gradient is evaluated.
    """
    def __init__(self, cmf):
        if getattr(cmf, '_ao_clusters', None) is None:
            raise RuntimeError('the scanner rebuilds the clusters from AO labels: define them with set_clusters_from_ao')
        self.base = cmf
        self.mol = cmf.mol
        self.verbose = cmf.verbose
        self.stdout = cmf.stdout
        self.atmlst = None
        self._mf_scanner = cmf.mf.as_scanner()

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)
        old = self.base
        mf = self._mf_scanner
        mf(mol)
        cmf = old.__class__(mf)
        for key in _SETTINGS:
            setattr(cmf, key, getattr(old, key))
        cmf.set_clusters_from_ao(*old._ao_clusters)
        if old.converged:
            # start from the optimized orbitals of the previous geometry
            cmf.set_mo_coeffs(_carry_orbitals(old.get_mo_coeffs(), cmf.S))
        if old.sa_states is not None:
            cmf.state_average_([st['fspace'] for st in old.sa_states], old.sa_weights)
        e_tot = cmf.kernel()
        de = kernel(cmf, self.atmlst, verbose=0)
        self.base, self.mol = cmf, mol
        return e_tot, de
