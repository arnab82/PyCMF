import numpy as np
import scipy.linalg
import pyscf
from pyscf import gto, scf, fci, ao2mo, lib, lo
from pyscf.lib import logger

from .helpers import * 

class CMF(lib.StreamObject):
    r"""
    Class to organize CMF calculation

    The cluster mean-field (cMF) wavefunction is a product of cluster states,
    :math:`|\Psi\rangle = |I\rangle \otimes |J\rangle \otimes \cdots`, built from one set of
    spin-restricted orbitals. Everything that couples different clusters (and the doubly occupied
    core) is a mean-field interaction, so it is evaluated in the AO basis with the J/K builder of the
    PySCF mean-field object (``mf.get_jk``). Direct, in-core and density-fitted integrals all work,
    and the integrals of the full active space are never formed: only the small :math:`n_I^4`
    integrals of each cluster are transformed.

    Open-shell clusters are treated as in restricted open-shell cMF (RO-cMF, Bachhar & Mayhall,
    J. Phys. Chem. A 128, 9015 (2024)): each cluster is an ensemble over the :math:`M_s` components
    of its spin multiplet, so only spin-summed 1RDMs and spin-free 2RDMs enter the energy, and the
    embedding potential (J - K/2) is spin independent.

    Examples
    --------
    >>> cmf = CMF(mf)
    >>> cmf.set_Cdoc(C[:, :2])
    >>> cmf.set_Cact(C[:, 2:8], [[0,1,2], [3,4,5]], [(2,1), (1,2)])
    >>> cmf.set_Cvir(C[:, 8:])
    >>> e = cmf.kernel()
    """
    def __init__(self, mf:pyscf.scf.hf.SCF):
        self.mf = mf
        self.mol = mf.mol
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.nbas = mf.mol.nao_nr()
        
        self.Cfrz = np.zeros((self.nbas,0))   # frozen core - no orbital optimization
        self.Cdoc = np.zeros((self.nbas,0))   # doubly occupied - no correlation
        self.Cact = np.zeros((self.nbas,0))   # active
        self.Cvir = np.zeros((self.nbas,0))   # virtual - no correlation
        self.S = self.mf.get_ovlp()

        # Active space clustering
        self.clusters = []
        self.fspace = []

        # Active space integrals
        self.active_h0 = 0.0
        self.active_h1 = np.zeros((0,0))
        self.active_h2 = np.zeros((0,0,0,0))

        # Orbital optimization
        self.conv_tol        = 1e-10    # change in energy
        self.conv_tol_grad   = 1e-6     # norm of the orbital gradient
        self.max_cycle       = 200
        self.max_stepsize    = 0.5      # max norm of an orbital rotation step
        self.lbfgs_memory    = 20
        self.zero_intra_rots = True     # rotations within a cluster are redundant for FCI

        # CMF-CI (cluster states at fixed orbitals)
        self.conv_tol_rdm1   = 1e-8     # change in the cluster 1RDMs
        self.max_cycle_rdm1  = 100
        self.sequential      = True     # update the embedding after each cluster (Gauss-Seidel)
        self.ci_conv_tol     = 1e-12    # cluster FCI
        self.ci_max_cycle    = 200

        # Results
        self.e_tot = None
        self.mo_coeff = None
        self.converged = False

        # State averaging, see state_average_
        self.sa_weights = None
        self.sa_states = None
        self.e_states = None
        self._state = 0

    def set_Cfrz(self, C:np.ndarray):
        """Define the frozen orbitals which are not optimized. 
        """
        self.Cfrz = C
        if len(self.Cfrz.shape) == 1:
            self.Cfrz.shape = (self.Cfrz.shape[0], 1)
    
    def set_Cdoc(self, C:np.ndarray):
        """Define the doubly occupied orbitals.
        These represent a closed shell cluster.
        """
        self.Cdoc = C
        if len(C.shape) == 1:
            self.Cdoc.shape = (C.shape[0], 1)
    
    def set_Cvir(self, C:np.ndarray):
        """Define the completely virtual orbitals.
        """
        self.Cvir = C
        if len(C.shape) == 1:
            self.Cvir.shape = (C.shape[0], 1)

    def set_Cact(self, C:np.ndarray, clusters:list[list[int]], fspace:list[tuple[int,int]]):
        """Define the active orbitals, with the associated clustering.
        
        Parameters
        ----------
        C: np.ndarray
            MO Coefficients of active space
        clusters : list
            A list of lists indicating which MOs are in each cluster.  
        fspace : list 
            A list of tuples indicating how many alpha and beta electrons are in each cluster 
        Returns
        -------
        None
        """
        assert len(clusters) == len(fspace)
        assert sorted(sum(clusters, [])) == list(range(C.shape[1])), "each active orbital must be in exactly one cluster"

        #   Get data
        self.Cact = C
        self.clusters = clusters 
        self.fspace   = fspace 
        self.cluster_energies:list[float] = [0.0 for i in clusters]
        self.cluster_1rdm = None    # spin summed 1RDM of each cluster, None -> guess from self.mf
        self.cluster_2rdm = [None for i in clusters]   # spin free 2RDM of each cluster
        self.ci = [None for i in clusters]
        self.roots = [1 for i in clusters]
        self.sa_weights = None
        self.sa_states = None
        self.e_states = None
        self._state = 0

    def state_average_(self, fspaces:list[list[tuple[int,int]]], weights=None):
        """Optimize the orbitals for the weighted average energy of several product states.

        Each state has its own cluster states, set by its `fspace`, and all states share the orbitals.
        For a mixed-valence dimer, averaging the two valence-trapped configurations, e.g.
        `[[(5,1), (5,0)], [(5,0), (5,1)]]`, gives orbitals that are not biased towards either one.
        `cmf_ci`, `orbital_gradient` and `kernel` then work with the average energy (`e_tot`) and
        the energy of each state is in `e_states`. Use `select_state(k)` to get the RDMs of state `k`.

        Parameters
        ----------
        fspaces : list
            For each state, a list of tuples indicating how many alpha and beta electrons are in each cluster
        weights : list
            Weight of each state, equal weights by default
        """
        nelec = [sum(na+nb for na, nb in f) for f in fspaces]
        assert all(len(f) == len(self.clusters) for f in fspaces)
        assert len(set(nelec)) == 1, "all states need the same number of electrons"
        if weights is None:
            weights = np.ones(len(fspaces))
        self.sa_weights = np.asarray(weights, dtype=float) / np.sum(weights)
        self.sa_states = [dict(fspace=[tuple(fi) for fi in f],
                               ci=[None for i in self.clusters],
                               cluster_1rdm=None,
                               cluster_2rdm=[None for i in self.clusters],
                               cluster_energies=[0.0 for i in self.clusters]) for f in fspaces]
        self.e_states = [None for f in fspaces]
        self._state = 0
        for key, val in self.sa_states[0].items():
            setattr(self, key, val)
        return self

    def select_state(self, k:int) -> None:
        """Make state `k` of a state-averaged calculation the current one, e.g. for its RDMs or `analyze`
        """
        if k == self._state:
            return
        if self.sa_states is None:
            raise ValueError('no state averaging, see state_average_')
        for key in self.sa_states[self._state]:
            self.sa_states[self._state][key] = getattr(self, key)
        for key, val in self.sa_states[k].items():
            setattr(self, key, val)
        self._state = k

    def _states(self) -> list[tuple]:
        """(index, weight) of each state in the energy: a single state, or the state-averaged ones
        """
        if self.sa_states is None:
            return [(None, 1.0)]
        return list(enumerate(self.sa_weights))

    def rotate_mo_coeffs(self, U:np.ndarray) -> None:
        """Rotate the [doc, act, vir] orbitals, C -> C U
        """
        self.set_mo_coeffs(self.get_mo_coeffs() @ U)

    def get_mo_coeffs(self) -> np.ndarray:
        return np.hstack((self.Cdoc, self.Cact, self.Cvir)) 

    def set_mo_coeffs(self, C:np.ndarray) -> None:
        """Overwrite the [doc, act, vir] orbitals, keeping the size of each space
        """
        ndoc = self.Cdoc.shape[1]
        nact = self.Cact.shape[1]
        self.Cdoc = C[:, :ndoc]
        self.Cact = C[:, ndoc:ndoc+nact]
        self.Cvir = C[:, ndoc+nact:]
    
    def lowdin(self) -> np.ndarray:
        """
        Get symetrically orthogonalized localized basis
        """
        print(" Form lowdin orthogonalized orbitals")
        #forming S^-1/2 to transform to A and B block.
        sal, svec = np.linalg.eigh(self.S)
        idx = sal.argsort()[::-1]
        sal = sal[idx]
        svec = svec[:, idx]
        sal = sal**-0.5
        sal = np.diagflat(sal)
        X = svec @ sal @ svec.T
        return X

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('frozen = %d  doc = %d  act = %d  vir = %d',
                 self.Cfrz.shape[1], self.Cdoc.shape[1], self.Cact.shape[1], self.Cvir.shape[1])
        for i, (ci, fi) in enumerate(zip(self.clusters, self.fspace)):
            log.info('cluster %3d  (na, nb) = %-8s orbitals = %s', i, tuple(fi), list(ci))
        log.info('conv_tol = %g  conv_tol_grad = %g  max_cycle = %d', self.conv_tol, self.conv_tol_grad, self.max_cycle)
        log.info('conv_tol_rdm1 = %g  max_cycle_rdm1 = %d  sequential = %s',
                 self.conv_tol_rdm1, self.max_cycle_rdm1, self.sequential)
        if self.sa_states is not None:
            for k, (w, st) in enumerate(zip(self.sa_weights, self.sa_states)):
                log.info('state %d  weight %.3f  fspace = %s', k, w, st['fspace'])
        nelec = 2*(self.Cfrz.shape[1] + self.Cdoc.shape[1]) + sum(na+nb for na, nb in self.fspace)
        if nelec != self.mol.nelectron:
            log.warn('CMF has %d electrons, but the molecule has %d', nelec, self.mol.nelectron)
        return self

    def build_active_integrals(self) -> None:
        """
        Build effective integrals for all active space clusters 
        """
        mol = self.mf.mol

        h0 = self.mf.energy_nuc()
        h1  = self.mf.get_hcore()

        nbas = self.nbas
        nact = self.Cact.shape[1]


        # Build AO 1rdm (spin summed)
        rdm1_embed = self.get_core_dm()

        j, k = self.mf.get_jk(self.mf.mol, rdm1_embed, hermi=1)

        h0 += np.einsum('ij,ji->', rdm1_embed, h1 + .5*j - .25*k)
        
        h1 += j - .5*k;

        # now rotate to MO basis
        h1 = self.Cact.T @ h1 @ self.Cact 
        h2 = pyscf.ao2mo.kernel(mol, self.Cact, aosym="s4", compact=False)
        h2.shape = (nact, nact, nact, nact)

        self.active_h0 = h0
        self.active_h1 = h1
        self.active_h2 = h2
        return

    def extract_cluster_integrals(self, i) -> (float, np.ndarray, np.ndarray):
        """
        Build effective integrals for cluster `i`
        """
        mol = self.mf.mol
        
        ci = self.clusters[i]

        h0  = self.active_h0
        h1  = self.active_h1[ci, :][:, ci]
        h2  = self.active_h2[ci,:,:,:][:,ci,:,:][:,:,ci,:][:, :, :, ci]

        return h0, h1, h2

    def do_local_casci(self, i) -> None:
        """
        Solve local problem for cluster i
        """
        
        h0, h1, h2 = self.extract_cluster_integrals(i)

        norb_i = h1.shape[1]

        cisolver = fci.direct_spin1.FCI()
        efci, ci = cisolver.kernel(h1, h2, norb_i, self.fspace[i], ecore=h0, nroots=1, verbose=100)

        self.cluster_energies[i] = efci

        fci_dim = ci.shape[0]*ci.shape[1]
        # (d1a, d1b)  = cisolver.make_rdm1s(ci, norb_i, self.fspace[i])
        
        # d1 = cisolver.make_rdm1(ci, norb_i, nelec)
        (d1a, d1b), (d2aa, d2ab, d2bb)  = cisolver.make_rdm12s(ci, norb_i, self.fspace[i])
        
        d1 = d1a + d1b

        print(" PYSCF 1RDM: ")
        matrix_print(d1a)
        
        occs = np.linalg.eig(d1)[0]

        print(" Occupation Numbers:")
        [print("%4i %12.8f"%(i,occs[i])) for i in range(len(occs))]
        print(" FCI:        %12.8f Dim:%6d"%(efci,fci_dim))
        
    #
    #   AO basis CMF
    #
    def get_core_dm(self) -> np.ndarray:
        """AO density (spin summed) of the frozen and doubly occupied orbitals
        """
        Ccore = np.hstack((self.Cfrz, self.Cdoc))
        return 2 * Ccore @ Ccore.T
        
    def get_cluster_dms(self, rdm1s=None) -> list[np.ndarray]:
        """AO density (spin summed) of each cluster
        """
        if rdm1s is None:
            rdm1s = self.cluster_1rdm
        dms = []
        for ci, d in zip(self.clusters, rdm1s):
            Ci = self.Cact[:, ci]
            dms.append(Ci @ d @ Ci.T)
        return dms

    def get_veff(self, rdm1s=None) -> (np.ndarray, list[np.ndarray]):
        """
        Mean-field potential, J - K/2, of the core and of each cluster in the AO basis

        Returns
        -------
        vcore : np.ndarray
        vclusters : list
        """
        if rdm1s is None:
            rdm1s = self.cluster_1rdm
        orbs = [np.hstack((self.Cfrz, self.Cdoc))]
        occs = [np.full(orbs[0].shape[1], 2.0)]
        for i, d in enumerate(rdm1s):
            c, n = self._cluster_natural_orbitals(i, d)
            orbs.append(c)
            occs.append(n)
        veff = self._get_veff(orbs, occs)
        return veff[0], list(veff[1:])

    def _cluster_natural_orbitals(self, i, rdm1) -> (np.ndarray, np.ndarray):
        """AO coefficients and occupations of the natural orbitals of cluster `i`
        """
        n, u = np.linalg.eigh(rdm1)
        return self.Cact[:, self.clusters[i]] @ u, np.maximum(n, 0)

    def _get_veff(self, orbs:list[np.ndarray], occs:list[np.ndarray]) -> np.ndarray:
        """
        J - K/2 for the AO densities C diag(n) C^T of each (C, n) in zip(orbs, occs)

        The orbitals are attached to the densities, so that density-fitted K is built from
        them at a cost proportional to the number of orbitals instead of the number of AOs
        """
        nmo = max(c.shape[1] for c in orbs)
        if nmo == 0:
            return np.zeros((len(orbs), self.nbas, self.nbas))
        mo_coeff = np.zeros((len(orbs), self.nbas, nmo))
        mo_occ = np.zeros((len(orbs), nmo))
        for i, (c, n) in enumerate(zip(orbs, occs)):
            mo_coeff[i, :, :c.shape[1]] = c
            mo_occ[i, :c.shape[1]] = n
        dms = lib.einsum('xpi,xi,xqi->xpq', mo_coeff, mo_occ, mo_coeff)
        dms = lib.tag_array(dms, mo_coeff=mo_coeff, mo_occ=mo_occ)
        vj, vk = self.mf.get_jk(self.mol, dms, hermi=1)
        return vj - .5*vk

    def ao2mo(self, mo_coeffs) -> np.ndarray:
        """
        Two-electron integrals (pq|rs) for four sets of orbitals, using the same
        integrals (density fitted, in-core or direct) as `self.mf`
        """
        mf = self.mf
        shape = [c.shape[1] for c in mo_coeffs]
        if hasattr(getattr(mf, 'with_df', None), 'ao2mo'):
            eri = mf.with_df.ao2mo(mo_coeffs, compact=False)
        elif getattr(mf, '_eri', None) is not None:
            eri = ao2mo.incore.general(mf._eri, mo_coeffs, compact=False)
        else:
            eri = ao2mo.outcore.general_iofree(self.mol, mo_coeffs, compact=False)
        return eri.reshape(shape)

    def get_cluster_eris(self, eris_ao=None) -> list[np.ndarray]:
        """(tu|vw) for the orbitals of each cluster
        """
        eris = []
        for i, ci in enumerate(self.clusters):
            Ci = self.Cact[:, ci]
            if eris_ao is None:
                eris.append(self.ao2mo((Ci, Ci, Ci, Ci)))
            else:
                eris.append(np.einsum('mt,muvw->tuvw', Ci, eris_ao[i]))
        return eris

    def get_cluster_eris_ao(self) -> list[np.ndarray]:
        """(mu u|v w) with the first index in the AO basis and the others in the orbitals of each cluster
        """
        if hasattr(getattr(self.mf, 'with_df', None), 'loop'):
            # density fitting: all clusters in one pass over the 3-index integrals
            eris_ao = [np.zeros((self.nbas,) + (len(ci),)*3) for ci in self.clusters]
            for eri1 in self.mf.with_df.loop():
                Lpa = lib.einsum('Lpq,qa->Lpa', lib.unpack_tril(eri1), self.Cact)
                for i, ci in enumerate(self.clusters):
                    Lvw = lib.einsum('pv,Lpw->Lvw', self.Cact[:, ci], Lpa[:, :, ci])
                    eris_ao[i] += lib.einsum('Lpu,Lvw->puvw', Lpa[:, :, ci], Lvw)
            return eris_ao
        eye = np.eye(self.nbas)
        eris_ao = []
        for ci in self.clusters:
            Ci = self.Cact[:, ci]
            eris_ao.append(self.ao2mo((eye, Ci, Ci, Ci)))
        return eris_ao

    def get_active_eris(self) -> np.ndarray:
        """(pq|rs) for all active orbitals
        """
        return self.ao2mo((self.Cact,) * 4)

    def _get_active_veff(self, i, rdm1, eri_act) -> np.ndarray:
        """J - K/2 of the density `rdm1` of cluster `i`, in the basis of all active orbitals
        """
        ci = self.clusters[i]
        vj = np.einsum('pqvw,vw->pq', eri_act[:, :, ci, :][:, :, :, ci], rdm1)
        vk = np.einsum('pwvq,vw->pq', eri_act[:, ci, :, :][:, :, ci, :], rdm1)
        return vj - .5*vk

    def get_cluster_veff(self) -> list[np.ndarray]:
        """J - K/2 (AO) of each cluster density, in one J/K build
        """
        if len(self.clusters) == 0:
            return []
        return list(self._get_veff(*zip(*[self._cluster_natural_orbitals(i, d) for i, d in enumerate(self.cluster_1rdm)])))

    def set_clusters_from_ao(self, ao_labels:list[list[str]], fspace:list[tuple[int,int]], method='meta_lowdin') -> None:
        """Define the clusters directly from atomic orbitals.

        The orbitals of cluster `I` are the orthogonalized AOs matching `ao_labels[I]`, e.g. `['0 Fe 3d']`.
        The other orthogonalized AOs are split into doubly occupied and virtual orbitals by diagonalizing the
        density of `self.mf` in that space. No mean-field orbitals have to be localized, and the charge and spin
        of each cluster come only from `fspace`. This is convenient for mixed-valence or antiferromagnetic
        systems, where the mean-field orbitals are delocalized or spin polarized. The orbital optimization
        (`kernel`) then relaxes the atomic orbitals.

        Parameters
        ----------
        ao_labels : list
            A list of AO labels (as in `mol.search_ao_label`) for each cluster
        fspace : list
            A list of tuples indicating how many alpha and beta electrons are in each cluster
        method : str
            Orthogonalization of the AOs, see `pyscf.lo.orth_ao`
        """
        assert self.Cfrz.shape[1] == 0, "frozen orbitals are not supported here"
        X = lo.orth_ao(self.mol, method, s=self.S)

        clusters = []
        idx_act = []
        for labels in ao_labels:
            idx = list(self.mol.search_ao_label(labels))
            clusters.append(list(range(len(idx_act), len(idx_act) + len(idx))))
            idx_act += idx
        assert len(set(idx_act)) == len(idx_act), "the same AO is in more than one cluster"
        idx_rest = [i for i in range(self.nbas) if i not in idx_act]

        nelec_act = sum(na+nb for na, nb in fspace)
        assert (self.mol.nelectron - nelec_act) % 2 == 0
        ndoc = (self.mol.nelectron - nelec_act) // 2

        dm = np.asarray(self.mf.make_rdm1())
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        SC = self.S @ X[:, idx_rest]
        occ, v = np.linalg.eigh(SC.T @ dm @ SC)
        Crest = X[:, idx_rest] @ v[:, ::-1]

        self.set_Cdoc(Crest[:, :ndoc])
        self.set_Cact(X[:, idx_act], clusters, fspace)
        self.set_Cvir(Crest[:, ndoc:])

    def get_init_guess_rdm1(self) -> list[np.ndarray]:
        """
        Guess the cluster 1RDMs by projecting the density of `self.mf` onto each cluster,
        or use uniform occupations if `self.mf` has no orbitals
        """
        dm = None
        if getattr(self.mf, 'mo_coeff', None) is not None:
            dm = np.asarray(self.mf.make_rdm1())
            if dm.ndim == 3:
                dm = dm[0] + dm[1]
        rdm1s = []
        for ci, (na, nb) in zip(self.clusters, self.fspace):
            if dm is None:
                rdm1s.append(np.eye(len(ci)) * (na+nb) / len(ci))
            else:
                SC = self.S @ self.Cact[:, ci]
                rdm1s.append(SC.T @ dm @ SC)
        return rdm1s

    def get_cluster_hamiltonian(self, i, dm_env, veff_env, hcore=None) -> (float, np.ndarray):
        """
        One-body part of the Hamiltonian for cluster `i` in the mean field of the core and other clusters

        Parameters
        ----------
        dm_env : np.ndarray
            AO density (spin summed) of everything outside cluster `i`
        veff_env : np.ndarray
            J - K/2 of `dm_env`

        Returns
        -------
        h0 : float
            Energy of the environment
        h1 : np.ndarray
            Dressed one-electron integrals in the orbitals of cluster `i`
        """
        if hcore is None:
            hcore = self.mf.get_hcore()
        Ci = self.Cact[:, self.clusters[i]]
        h0 = self.mf.energy_nuc() + np.einsum('ij,ji->', dm_env, hcore + .5*veff_env)
        h1 = Ci.T @ (hcore + veff_env) @ Ci
        return h0, h1

    def solve_cluster(self, i, h0, h1, h2):
        """
        Solve the FCI problem for cluster `i`

        Returns
        -------
        e : float
        ci : np.ndarray
        d1 : np.ndarray
            spin summed 1RDM
        d2 : np.ndarray
            spin free 2RDM, d2[p,q,r,s] = <p^+ r^+ s q>
        """
        norb = h1.shape[0]
        nelec = tuple(self.fspace[i])
        solver = fci.direct_spin1.FCISolver(self.mol)
        solver.conv_tol = self.ci_conv_tol
        solver.max_cycle = self.ci_max_cycle
        ci0 = self.ci[i]
        if ci0 is not None and ci0.size != fci.cistring.num_strings(norb, nelec[0]) * fci.cistring.num_strings(norb, nelec[1]):
            ci0 = None
        e, ci = solver.kernel(h1, h2, norb, nelec, ci0=ci0, ecore=h0)
        d1, d2 = solver.make_rdm12(ci, norb, nelec)
        return e, ci, d1, d2

    def get_core_veff(self) -> np.ndarray:
        """J - K/2 (AO) of the frozen and doubly occupied orbitals
        """
        Ccore = np.hstack((self.Cfrz, self.Cdoc))
        return self._get_veff([Ccore], [np.full(Ccore.shape[1], 2.0)])[0]

    def cmf_ci(self, eri_act=None, vcore=None) -> float:
        """
        Converge the cluster states for the current orbitals (CMF-CI)

        Each cluster is solved with FCI in the mean field (J - K/2) of the core and the other clusters,
        until the cluster 1RDMs stop changing. The core potential is built once in the AO basis; the
        interactions between clusters only need the integrals of the active orbitals, so the sweeps
        do not touch the AO basis. With `state_average_` every state is converged and the average
        energy is returned.

        Parameters
        ----------
        eri_act : np.ndarray
            (pq|rs) of all active orbitals, computed if not given
        vcore : np.ndarray
            J - K/2 (AO) of the core, computed if not given

        Returns
        -------
        e_tot : float
            CMF energy
        """
        if eri_act is None:
            eri_act = self.get_active_eris()
        if vcore is None:
            vcore = self.get_core_veff()
        e_tot = 0.0
        for k, w in self._states():
            if k is None:
                e_tot = self._cmf_ci(eri_act, vcore)
            else:
                self.select_state(k)
                self.e_states[k] = self._cmf_ci(eri_act, vcore)
                e_tot += w * self.e_states[k]
        if self.sa_states is not None:
            self.select_state(0)
        self.e_tot = e_tot
        return e_tot

    def _cmf_ci(self, eri_act, vcore) -> float:
        """CMF-CI for the current state
        """
        log = logger.new_logger(self)
        nclusters = len(self.clusters)
        if self.cluster_1rdm is None:
            self.cluster_1rdm = self.get_init_guess_rdm1()

        hcore = self.mf.get_hcore()
        e_core = self.mf.energy_nuc() + np.einsum('ij,ji->', self.get_core_dm(), hcore + .5*vcore)
        h_act = self.Cact.T @ (hcore + vcore) @ self.Cact

        eris = [eri_act[np.ix_(ci, ci, ci, ci)] for ci in self.clusters]
        vact = [self._get_active_veff(i, d, eri_act) for i, d in enumerate(self.cluster_1rdm)]
        v_sum = sum(vact) + np.zeros_like(h_act)
        dm_act = self.make_rdm1()

        e_tot = 0.0
        conv = False
        for it in range(self.max_cycle_rdm1):
            d_err = 0.0
            for i, ci in enumerate(self.clusters):
                idx = np.ix_(ci, ci)
                v_env = v_sum - vact[i]
                dm_env = dm_act.copy()
                dm_env[idx] = 0
                h0 = e_core + np.einsum('ij,ji->', dm_env, h_act + .5*v_env)
                e, civec, d1, d2 = self.solve_cluster(i, h0, (h_act + v_env)[idx], eris[i])

                d_err = max(d_err, np.linalg.norm(d1 - self.cluster_1rdm[i]))
                self.ci[i] = civec
                self.cluster_energies[i] = e
                self.cluster_1rdm[i] = d1
                self.cluster_2rdm[i] = d2
                dm_act[idx] = d1

                if self.sequential:
                    v_i = self._get_active_veff(i, d1, eri_act)
                    v_sum += v_i - vact[i]
                    vact[i] = v_i

            if not self.sequential and nclusters > 0:
                vact = [self._get_active_veff(i, d, eri_act) for i, d in enumerate(self.cluster_1rdm)]
                v_sum = sum(vact)

            e_last = e_tot
            e_tot = e_core + np.einsum('ij,ji->', dm_act, h_act + .5*v_sum)
            for i, ci in enumerate(self.clusters):
                e_tot += (.5*np.einsum('tuvw,tuvw->', eris[i], self.cluster_2rdm[i])
                          - .5*np.einsum('ij,ji->', self.cluster_1rdm[i], vact[i][np.ix_(ci, ci)]))
            log.debug('CMF-CI iter %3d  E = %.12f  dE = %.3e  |dD| = %.3e', it, e_tot, e_tot - e_last, d_err)
            if d_err < self.conv_tol_rdm1:
                conv = True
                break
        if not conv:
            log.warn('CMF-CI not converged: |dD| = %.3e', d_err)

        if log.verbose >= logger.DEBUG:
            for i, ci in enumerate(self.clusters):
                ss = fci.spin_square(self.ci[i], len(ci), self.fspace[i])[0]
                log.debug('  cluster %3d  E = %.12f  <S^2> = %.4f', i, self.cluster_energies[i], ss)
        return e_tot

    def energy(self, veff=None, eris=None, hcore=None) -> float:
        """
        Compute current CMF energy

        The mean-field energy of the total density, with the mean-field energy inside each
        cluster replaced by the exact one from its 2RDM:

        E = E_nuc + tr(D h) + 1/2 tr(D (J-K/2)[D]) + sum_I [1/2 (tu|vw) G^I_tuvw - 1/2 tr(D_I (J-K/2)[D_I])]
        """
        if hcore is None:
            hcore = self.mf.get_hcore()
        if veff is None:
            veff = self.get_veff()
        if eris is None:
            eris = self.get_cluster_eris()
        vcore, vclusters = veff
        dms = self.get_cluster_dms()

        dm = self.get_core_dm() + sum(dms)
        v = vcore + sum(vclusters)
        e = self.mf.energy_nuc() + np.einsum('ij,ji->', dm, hcore + .5*v)
        for dmi, vi, eri, d2 in zip(dms, vclusters, eris, self.cluster_2rdm):
            e += .5*np.einsum('tuvw,tuvw->', eri, d2) - .5*np.einsum('ij,ji->', dmi, vi)
        return e

    def get_occ(self) -> np.ndarray:
        """Spin summed 1RDM in the [doc, act, vir] orbital basis
        """
        ndoc = self.Cdoc.shape[1]
        nmo = ndoc + self.Cact.shape[1] + self.Cvir.shape[1]
        occ = np.zeros((nmo, nmo))
        occ[:ndoc, :ndoc] = 2*np.eye(ndoc)
        for ci, d in zip(self.clusters, self.cluster_1rdm):
            idx = ndoc + np.asarray(ci)
            occ[np.ix_(idx, idx)] = d
        return occ

    def energy_deriv(self, veff=None, eris_ao=None) -> np.ndarray:
        """
        Derivative of the CMF energy with respect to the [doc, act, vir] orbital coefficients, dE/dC (AO x MO),
        at fixed cluster RDMs

        dE/dC = 2 F C n  -  2 (J-K/2)[D_I] C_I d_I  +  2 (mu u|v w) G^I_tuvw
        """
        if veff is None:
            veff = self.get_veff()
        if eris_ao is None:
            eris_ao = self.get_cluster_eris_ao()
        vcore, vclusters = veff
        C = self.get_mo_coeffs()
        fock = self.mf.get_hcore() + vcore + sum(vclusters)
        dEdC = 2 * fock @ C @ self.get_occ()

        ndoc = self.Cdoc.shape[1]
        for i, ci in enumerate(self.clusters):
            Ci = self.Cact[:, ci]
            x = np.einsum('muvw,tuvw->mt', eris_ao[i], self.cluster_2rdm[i])
            dEdC[:, ndoc + np.asarray(ci)] += 2*(x - vclusters[i] @ Ci @ self.cluster_1rdm[i])
        return dEdC

    def get_gen_fock(self, veff=None, eris_ao=None) -> np.ndarray:
        """Generalized Fock matrix, F = 1/2 C^T dE/dC, in the [doc, act, vir] orbital basis
        """
        return .5 * self.get_mo_coeffs().T @ self.energy_deriv(veff, eris_ao)

    def _average_deriv(self, vcore, eris_ao, with_hdiag=False) -> (np.ndarray, np.ndarray):
        """dE/dC averaged over the states (one state without `state_average_`), and optionally the
        averaged diagonal Hessian guess, for the current orbitals and cluster states
        """
        dEdC = 0
        hdiag = None
        for k, w in self._states():
            if k is not None:
                self.select_state(k)
            veff = (vcore, self.get_cluster_veff())
            d = self.energy_deriv(veff, eris_ao)
            dEdC = dEdC + w * d
            if with_hdiag:
                h = w * self.approx_hessian_diag(veff, d)
                hdiag = h if hdiag is None else hdiag + h
        if self.sa_states is not None:
            self.select_state(0)
        return dEdC, hdiag

    def rotation_mask(self) -> np.ndarray:
        """
        Lower triangular mask of the non-redundant orbital rotations, kappa_pq with p > q,
        in the [doc, act, vir] orbital basis
        """
        ndoc = self.Cdoc.shape[1]
        nact = self.Cact.shape[1]
        nmo = ndoc + nact + self.Cvir.shape[1]
        space = np.full(nmo, -1)
        space[ndoc+nact:] = -2
        for i, ci in enumerate(self.clusters):
            space[ndoc + np.asarray(ci)] = i
        mask = space[:, None] != space[None, :]
        if not self.zero_intra_rots:
            mask |= (space[:, None] == space[None, :]) & (space[:, None] >= 0)
        return np.tril(mask, -1)

    def orbital_gradient(self, veff=None) -> np.ndarray:
        """
        Compute current CMF orbital gradient

        g_pq = dE/dkappa_pq = 2 (F_pq - F_qp) for C -> C exp(kappa), as an antisymmetric matrix
        in the [doc, act, vir] orbital basis with the redundant rotations set to zero.
        With `state_average_` this is the gradient of the average energy.
        """
        if self.sa_states is None:
            F = self.get_gen_fock(veff)
        else:
            F = .5 * self.get_mo_coeffs().T @ self._average_deriv(self.get_core_veff(), self.get_cluster_eris_ao())[0]
        mask = self.rotation_mask()
        return 2*(F - F.T) * (mask | mask.T)

    def approx_hessian_diag(self, veff=None, dEdC=None) -> np.ndarray:
        """
        Diagonal orbital Hessian guess, 2 (n_p f_qq + n_q f_pp - F_pp - F_qq), where f is the
        mean-field Fock matrix and F the generalized Fock matrix
        """
        if veff is None:
            veff = self.get_veff()
        if dEdC is None:
            dEdC = self.energy_deriv(veff)
        vcore, vclusters = veff
        C = self.get_mo_coeffs()
        f = np.diag(C.T @ (self.mf.get_hcore() + vcore + sum(vclusters)) @ C)
        n = np.diag(self.get_occ())
        F = .5 * np.einsum('mp,mp->p', C, dEdC)
        return 2*(n[:, None]*f[None, :] + n[None, :]*f[:, None] - F[:, None] - F[None, :])

    def kernel(self) -> float:
        """
        Optimize the orbitals and the cluster states (orbital optimized CMF)

        The orbitals are C = C0 exp(kappa). Every energy evaluation converges CMF-CI, so the cluster
        states are stationary and the exact gradient with respect to kappa only needs the cluster RDMs.
        kappa is optimized with L-BFGS, starting from a diagonal Hessian guess, with a backtracking line search.

        Returns
        -------
        e_tot : float
            CMF energy
        """
        log = logger.new_logger(self)
        self.dump_flags()

        C0 = self.get_mo_coeffs()
        nmo = C0.shape[1]
        mask = self.rotation_mask()

        def unpack(x):
            K = np.zeros((nmo, nmo))
            K[mask] = x
            return K - K.T

        def energy_and_grad(x, with_hdiag=False):
            K = unpack(x)
            self.set_mo_coeffs(C0 @ scipy.linalg.expm(K))
            vcore = self.get_core_veff()
            e = self.cmf_ci(self.get_active_eris(), vcore)
            dEdC, hdiag = self._average_deriv(vcore, self.get_cluster_eris_ao(), with_hdiag)
            # chain rule through exp(K): the adjoint of the Frechet derivative of exp at K is the one at K^T
            dEdK = scipy.linalg.expm_frechet(K.T, C0.T @ dEdC, compute_expm=False)
            if with_hdiag:
                return e, (dEdK - dEdK.T)[mask], hdiag
            return e, (dEdK - dEdK.T)[mask]

        x = np.zeros(np.count_nonzero(mask))
        e, g, hdiag = energy_and_grad(x, with_hdiag=True)
        hdiag = np.maximum(abs(hdiag[mask]), 0.1)
        log.info('ooCMF iter %3d  E = %.12f  |g| = %.3e', 0, e, np.linalg.norm(g))

        s_hist, y_hist = [], []
        self.converged = np.linalg.norm(g) < self.conv_tol_grad
        for it in range(1, self.max_cycle+1):
            if self.converged:
                break
            step = -_lbfgs_direction(g, s_hist, y_hist, hdiag)
            if np.dot(step, g) >= 0:
                s_hist, y_hist = [], []
                step = -g / hdiag
            if np.linalg.norm(step) > self.max_stepsize:
                step *= self.max_stepsize / np.linalg.norm(step)

            # Backtracking line search
            alpha = 1.0
            for ls in range(10):
                e_new, g_new = energy_and_grad(x + alpha*step)
                if e_new <= e + 1e-4 * alpha * np.dot(g, step):
                    break
                alpha *= .5
            else:
                if not s_hist:
                    log.warn('ooCMF line search failed')
                    energy_and_grad(x)
                    break
                log.debug('ooCMF line search failed, resetting L-BFGS')
                s_hist, y_hist = [], []
                e, g = energy_and_grad(x)
                continue

            s, y = alpha*step, g_new - g
            if np.dot(s, y) > 1e-12:
                s_hist.append(s)
                y_hist.append(y)
                if len(s_hist) > self.lbfgs_memory:
                    s_hist.pop(0)
                    y_hist.pop(0)

            de = e_new - e
            x, e, g = x + s, e_new, g_new
            log.info('ooCMF iter %3d  E = %.12f  dE = %.3e  |g| = %.3e  |step| = %.3e',
                     it, e, de, np.linalg.norm(g), np.linalg.norm(s))
            self.converged = np.linalg.norm(g) < self.conv_tol_grad and abs(de) < self.conv_tol

        if self.converged:
            log.note('ooCMF converged  E = %.12f', e)
        else:
            log.note('ooCMF not converged  E = %.12f  |g| = %.3e', e, np.linalg.norm(g))
        if self.sa_states is not None:
            for k, (w, ek) in enumerate(zip(self.sa_weights, self.e_states)):
                log.note('  state %d  weight %.3f  E = %.12f  fspace = %s', k, w, ek, self.sa_states[k]['fspace'])
        self.e_tot = e
        self.mo_coeff = np.hstack((self.Cfrz, self.get_mo_coeffs()))
        return e

    def make_rdm1s(self, state=None, ncas=None, nelecas=None, root=0, spin_average=False, ao_repr=False):
        """Returns the aa and bb blocks of the spin 1RDM.

        By default these are the densities of the product state with each cluster in the Ms sector
        given by `fspace`, e.g. for spin densities. With `spin_average=True` each cluster is the ensemble
        over the Ms components of its spin multiplet that defines the RO-cMF energy, so the alpha and
        beta densities are both half of the spin summed density.

        Parameters
        ----------
        state : np.ndarray
            Unused, but kept to continuity with `mcscf.CASCI`
        ncas : int
            Unused, but kept to continuity with `mcscf.CASCI`
        nelecas : int
            Unused, but kept to continuity with `mcscf.CASCI`
        root : int
            State to read RDM for, default is 0.
        spin_average : bool
            Average each cluster over its Ms components
        ao_repr : bool
            Return the densities of the whole system (frozen and doubly occupied orbitals included) in the AO basis

        Returns
        -------
        tuple
            A tuple of the aa and bb blocks of the spin 1RDM.
            E.g. (dm_a, dm_b)
        """
        nact = self.Cact.shape[1]
        dm_a = np.zeros((nact, nact))
        dm_b = np.zeros((nact, nact))
        for i, ci in enumerate(self.clusters):
            if spin_average:
                da = db = .5*self.cluster_1rdm[i]
            else:
                da, db = fci.direct_spin1.make_rdm1s(self.ci[i], len(ci), self.fspace[i])
            dm_a[np.ix_(ci, ci)] = da
            dm_b[np.ix_(ci, ci)] = db
        if ao_repr:
            Ccore = np.hstack((self.Cfrz, self.Cdoc))
            dm_a = Ccore @ Ccore.T + self.Cact @ dm_a @ self.Cact.T
            dm_b = Ccore @ Ccore.T + self.Cact @ dm_b @ self.Cact.T
        return (dm_a, dm_b)
    def make_rdm1(self, state=None, norb=None, nelec=None, link_index=None, ao_repr=False, **kwargs):
        """Spin summed 1RDM of the active space, or of the whole system in the AO basis if `ao_repr`
        """
        if ao_repr:
            return self.get_core_dm() + sum(self.get_cluster_dms())
        nact = self.Cact.shape[1]
        dm1 = np.zeros((nact, nact))
        for ci, d in zip(self.clusters, self.cluster_1rdm):
            dm1[np.ix_(ci, ci)] = d
        return dm1
    def make_rdm12s(self, state=None, norb=None, nelec=None, link_index=None, **kwargs):
        """Spin resolved 1RDMs and 2RDMs of the active space for the product state with each cluster
        in the Ms sector given by `fspace`

        The 2RDMs are dm2aa[p,q,r,s] = <p_a^+ r_a^+ s_a q_a>, dm2ab[p,q,r,s] = <p_a^+ r_b^+ s_b q_a>, ...
        (PySCF convention). Between clusters the product state gives the mean-field form.
        Note that the RO-cMF energy (`self.e_tot`) is the one of the Ms ensemble, see `make_rdm12`;
        the energy of this product state differs by the exchange between the cluster spin densities.

        Returns
        -------
        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)
        """
        dm1a, dm1b = self.make_rdm1s()
        dm2aa = np.einsum('pq,rs->pqrs', dm1a, dm1a) - np.einsum('ps,rq->pqrs', dm1a, dm1a)
        dm2bb = np.einsum('pq,rs->pqrs', dm1b, dm1b) - np.einsum('ps,rq->pqrs', dm1b, dm1b)
        dm2ab = np.einsum('pq,rs->pqrs', dm1a, dm1b)
        for i, ci in enumerate(self.clusters):
            d1, (d2aa, d2ab, d2bb) = fci.direct_spin1.make_rdm12s(self.ci[i], len(ci), self.fspace[i])
            idx = np.ix_(ci, ci, ci, ci)
            dm2aa[idx] = d2aa
            dm2ab[idx] = d2ab
            dm2bb[idx] = d2bb
        return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)
    def make_rdm12(self, state=None, norb=None, nelec=None, link_index=None, **kwargs):
        """Spin summed 1RDM and spin free 2RDM of the active space for the RO-cMF state,
        consistent with `self.e_tot`
        """
        dm1 = self.make_rdm1()
        
        # 2RDMs should be stored here as E2[i1,k1,j2,l2] (for PySCF purposes)
        # Between clusters the product state gives the mean-field form: Coulomb - exchange/2
        dm2 = np.einsum('pq,rs->pqrs', dm1, dm1) - .5*np.einsum('ps,rq->pqrs', dm1, dm1)
        for ci, d in zip(self.clusters, self.cluster_2rdm):
            dm2[np.ix_(ci, ci, ci, ci)] = d
        return dm1, dm2
    def trans_rdm1(self, statebra, stateket, norb, nelec, link_index=None, **kwargs):
        return self.trans_rdm12(statebra, stateket, norb, nelec, link_index, **kwargs)[0]
    def trans_rdm12(self, statebra, stateket, norb, nelec, link_index=None, **kwargs):
        raise NotImplementedError
    def make_rdm123(self, state, norb, nelec, link_index=None, **kwargs):
        # 3RDMs should be stored here as E3[i1,j2,k3,l3,m2,n1] (for PySCF purposes)
        raise NotImplementedError
    def _make_dm123(self, state, norb, nelec, **kwargs):
        r"""Note this function does NOT compute the standard density matrix.
        The density matrices are reordered to match the the fci.rdm.make_dm123
        function (used by NEVPT code).
        The returned "2rdm" is :math:`\langle p^\dagger q r^\dagger s\rangle`;
        The returned "3rdm" is :math:`\langle p^\dagger q r^\dagger s t^\dagger u\rangle`.
        """
        one_rdm, two_rdm, three_rdm = self.make_rdm123(state, norb, nelec, None, **kwargs)
        three_rdm = np.einsum("mkijln->ijklmn", three_rdm).copy()
        three_rdm += np.einsum("jk,lm,in->ijklmn", np.identity(norb), np.identity(norb), one_rdm)
        three_rdm += np.einsum("jk,miln->ijklmn", np.identity(norb), two_rdm)
        three_rdm += np.einsum("lm,kijn->ijklmn", np.identity(norb), two_rdm)
        three_rdm += np.einsum("jm,kinl->ijklmn", np.identity(norb), two_rdm)

        two_rdm = np.einsum("iklj->ijkl", two_rdm) + np.einsum("li,jk->ijkl", one_rdm, np.identity(norb))

        return one_rdm, two_rdm, three_rdm
    def make_rdm4(self,state,norb,nelec,**kwargs,):
        # should be written here as E4[i1,j2,k3,l4,p1,o2,n3,m4]
        # (for PySCF purposes)
        raise NotImplementedError

    def spin_square(self):
        """
        <S^2> of the product state with each cluster in the Ms sector given by `fspace`

        The transverse parts of S_I.S_J vanish for a product state, so
        <S^2> = sum_I <S_I^2> + 2 sum_{I<J} M_I M_J

        Returns
        -------
        ss : float
            <S^2> of the product state
        multip : float
            2S+1
        ss_clusters : list
            <S_I^2> of each cluster
        """
        ss_clusters = [fci.spin_square(self.ci[i], len(ci), self.fspace[i])[0] for i, ci in enumerate(self.clusters)]
        ms = [.5*(na - nb) for na, nb in self.fspace]
        ss = sum(ss_clusters) + sum(ms)**2 - sum(m**2 for m in ms)
        s = np.sqrt(ss + .25) - .5
        return ss, 2*s + 1, ss_clusters
    
    def analyze(self, verbose=None):
        """Print the cluster states, the populations and the dipole moment of the CMF wavefunction

        Returns
        -------
        pop_and_charges : tuple
            Meta-Lowdin populations and charges, see `pyscf.scf.uhf.mulliken_meta`
        dip : np.ndarray
            Dipole moment (Debye)
        """
        log = logger.new_logger(self, verbose)
        ss, multip, ss_clusters = self.spin_square()
        if self.sa_states is None:
            log.note('CMF energy = %.12f', self.e_tot)
        else:
            log.note('CMF energy of state %d = %.12f  (state average %.12f)', self._state, self.e_states[self._state], self.e_tot)
        log.note('cluster  (na, nb)      N       <S_I^2>   E (cluster in its embedding)')
        for i, ci in enumerate(self.clusters):
            d1 = self.cluster_1rdm[i]
            log.note('%5d    %-8s %8.4f %10.4f   %.10f', i, tuple(self.fspace[i]), np.trace(d1), ss_clusters[i], self.cluster_energies[i])
            log.info('         natural occupations: %s', ' '.join('%.4f' % n for n in np.linalg.eigvalsh(d1)[::-1]))
        log.note('product state <S^2> = %.4f  2S+1 = %.4f', ss, multip)

        dm = self.make_rdm1s(ao_repr=True)
        pop_and_charges = scf.uhf.mulliken_meta(self.mol, dm, verbose=log, s=self.S)
        dip = scf.hf.dip_moment(self.mol, dm[0] + dm[1], verbose=log)
        return pop_and_charges, dip

    def get_active_hamiltonian(self, clusters=None, eri_act=None, vcore=None) -> (float, np.ndarray, np.ndarray, np.ndarray):
        """
        Active-space Hamiltonian of the clusters with indices `clusters` (default: all), in the current orbitals.

        The frozen and doubly occupied orbitals, and the other clusters in their cMF states, are folded into
        h0 and h1: h1 gets their J - K/2 and h0 their energy, with the exact intra-cluster energy of the other
        clusters. So the cMF state of the selected clusters has the total cMF energy in this Hamiltonian, and
        solving it exactly adds the correlation between them in the mean field of the rest.
        With `state_average_` the other clusters enter with their state-averaged densities and energies.

        Returns
        -------
        h0 : float
        h1, h2 : np.ndarray
            One- and two-electron integrals, (pq|rs), in the orbitals of the selected clusters
        orbs : np.ndarray
            Indices (columns of `Cact`) of these orbitals, cluster after cluster
        """
        if clusters is None:
            clusters = range(len(self.clusters))
        clusters = list(clusters)
        others = [j for j in range(len(self.clusters)) if j not in clusters]
        if eri_act is None:
            eri_act = self.get_active_eris()
        if vcore is None:
            vcore = self.get_core_veff()

        hcore = self.mf.get_hcore()
        h0 = self.mf.energy_nuc() + np.einsum('ij,ji->', self.get_core_dm(), hcore + .5*vcore)
        h_act = self.Cact.T @ (hcore + vcore) @ self.Cact

        # 1RDMs and intra-cluster two-electron energies of the other clusters
        rdm1s = {j: 0 for j in others}
        e2 = {j: 0 for j in others}
        for k, w in self._states():
            if k is not None:
                self.select_state(k)
            for j in others:
                cj = self.clusters[j]
                rdm1s[j] = rdm1s[j] + w * self.cluster_1rdm[j]
                e2[j] += w * .5*np.einsum('tuvw,tuvw->', eri_act[np.ix_(cj, cj, cj, cj)], self.cluster_2rdm[j])
        if self.sa_states is not None:
            self.select_state(0)

        dm_env = np.zeros_like(h_act)
        v_env = np.zeros_like(h_act)
        for j in others:
            idx = np.ix_(self.clusters[j], self.clusters[j])
            vj = self._get_active_veff(j, rdm1s[j], eri_act)
            dm_env[idx] = rdm1s[j]
            v_env += vj
            h0 += np.einsum('ij,ji->', rdm1s[j], h_act[idx]) + e2[j] - .5*np.einsum('ij,ji->', rdm1s[j], vj[idx])
        h0 += .5*np.einsum('ij,ji->', dm_env, v_env)

        orbs = np.asarray(sum([list(self.clusters[i]) for i in clusters], []), dtype=int)
        h1 = (h_act + v_env)[np.ix_(orbs, orbs)]
        h2 = eri_act[np.ix_(orbs, orbs, orbs, orbs)]
        return h0, h1, h2, orbs

    def qdpt2(self, clusters=None, fspaces=None, **kwargs):
        """Quasi-degenerate PT2 for a few clusters in the mean field of the others, see `pycmf.pt2.qdpt2`
        """
        from .pt2 import qdpt2
        return qdpt2(self, clusters, fspaces, **kwargs)

    def pair_pt2(self, pairs=None, cutoff=None, **kwargs):
        """Pair-additive cMF-PT2 energy, see `pycmf.pt2.pair_pt2`
        """
        from .pt2 import pair_pt2
        return pair_pt2(self, pairs, cutoff, **kwargs)

    def pt2_mbe(self, order=2, cutoff=None, with_rdm=False, **kwargs):
        """cMF-PT2 energy (and 1RDM) from a many-body expansion over pairs and triples of clusters,
        see `pycmf.pt2.pt2_mbe`
        """
        from .pt2 import pt2_mbe
        return pt2_mbe(self, order, cutoff, with_rdm=with_rdm, **kwargs)

    def nevpt2(self, spin_average=True, **kwargs):
        """SC-NEVPT2 dynamic correlation (doubly occupied and virtual orbitals) of the current state,
        see `pycmf.mrpt.nevpt2`
        """
        from .mrpt import nevpt2
        return nevpt2(self, spin_average, **kwargs)

    def nevpt2_mbe(self, order=2, cutoff=None, **kwargs):
        """cMF-NEVPT2 from a many-body expansion over the clusters, see `pycmf.mrpt.nevpt2_mbe`
        """
        from .mrpt import nevpt2_mbe
        return nevpt2_mbe(self, order, cutoff, **kwargs)

    def export_fermicg(self, filename:str, clusters=None) -> None:
        """
        Write the active-space problem for post-cMF methods in FermiCG.jl (TPSCI, BST, PT2) to an npz file.

        `clusters` selects a subset of the clusters (e.g. a pair, to get the coupling between two metals);
        the other clusters are folded into the mean field, see `get_active_hamiltonian`.
        The active orbitals are reordered so that the orbitals of each cluster are contiguous. Keys:

        h0, h1, h2      `InCoreInts` in the current orbitals: energy of everything outside the selected
                        clusters (with nuclear repulsion), one-electron integrals dressed with its J - K/2,
                        and (pq|rs)
        cluster_index   cluster (1-based, in the order of `clusters`) of each active orbital
        fspace          (nclusters, 2) alpha and beta electrons of each cluster, the reference `FockConfig`
        fspace_states   (nstates, nclusters, 2) and `weights` of the states, see `state_average_`
        rdm1a, rdm1b    active-space 1RDMs for embedding the cluster bases; each cluster is an Ms ensemble,
                        so they are equal, and they are averaged over the states
        mo_coeff        AO coefficients of [frozen, doubly occupied, active (reordered), other clusters, virtual]
        ncore, nact     number of frozen and doubly occupied orbitals, and of active orbitals
        e_cmf           CMF energy (state average), and `e_states`

        In Julia::

            using NPZ, QCBase, RDM, InCoreIntegrals, FermiCG
            d = npzread("cmf.npz")
            ints = InCoreInts(first(d["h0"]), d["h1"], d["h2"])
            clusters = [MOCluster(i, findall(==(i), d["cluster_index"])) for i in 1:maximum(d["cluster_index"])]
            init_fspace = FockConfig([Tuple(d["fspace"][i, :]) for i in 1:size(d["fspace"], 1)])
            d1 = RDM1(d["rdm1a"], d["rdm1b"])
        """
        if self.cluster_1rdm is None:
            raise RuntimeError('run cmf_ci or kernel first')
        if clusters is None:
            clusters = range(len(self.clusters))
        clusters = list(clusters)
        h0, h1, h2, orbs = self.get_active_hamiltonian(clusters)
        others = [p for p in range(self.Cact.shape[1]) if p not in set(orbs)]
        cluster_index = np.concatenate([np.full(len(self.clusters[i]), n+1) for n, i in enumerate(clusters)])

        dm1 = 0
        for k, w in self._states():
            if k is not None:
                self.select_state(k)
            dm1 = dm1 + w * self.make_rdm1()
        if self.sa_states is None:
            fspace_states, weights, e_states = [self.fspace], [1.0], [self.e_tot]
        else:
            self.select_state(0)
            fspace_states, weights, e_states = [st['fspace'] for st in self.sa_states], self.sa_weights, self.e_states
        dm1 = dm1[np.ix_(orbs, orbs)]

        np.savez(filename, h0=np.float64(h0), h1=h1, h2=h2,
                 cluster_index=cluster_index.astype(np.int64),
                 fspace=np.asarray([self.fspace[i] for i in clusters], dtype=np.int64),
                 fspace_states=np.asarray([[f[i] for i in clusters] for f in fspace_states], dtype=np.int64),
                 weights=np.asarray(weights, dtype=float),
                 rdm1a=.5*dm1, rdm1b=.5*dm1,
                 mo_coeff=np.hstack((self.Cfrz, self.Cdoc, self.Cact[:, orbs], self.Cact[:, others], self.Cvir)),
                 ncore=np.int64(self.Cfrz.shape[1] + self.Cdoc.shape[1]),
                 nact=np.int64(len(orbs)),
                 e_cmf=np.float64(self.e_tot),
                 e_states=np.asarray(e_states, dtype=float))


def _lbfgs_direction(g, s_hist, y_hist, hdiag):
    """L-BFGS two-loop recursion, H^-1 g, with diag(hdiag) as the initial Hessian
    """
    q = g.copy()
    alphas = []
    for s, y in zip(reversed(s_hist), reversed(y_hist)):
        a = np.dot(s, q) / np.dot(y, s)
        q -= a * y
        alphas.append(a)
    r = q / hdiag
    for s, y, a in zip(s_hist, y_hist, reversed(alphas)):
        b = np.dot(y, r) / np.dot(y, s)
        r += s * (a - b)
    return r
