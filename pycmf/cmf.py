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
        dms = [self.get_core_dm()] + self.get_cluster_dms(rdm1s)
        vj, vk = self.mf.get_jk(self.mol, np.asarray(dms), hermi=1)
        veff = vj - .5*vk
        return veff[0], list(veff[1:])

    def ao2mo(self, mo_coeffs) -> np.ndarray:
        """
        Two-electron integrals (pq|rs) for four sets of orbitals, using the same
        integrals (density fitted, in-core or direct) as `self.mf`
        """
        mf = self.mf
        shape = [c.shape[1] for c in mo_coeffs]
        if getattr(mf, 'with_df', None) is not None:
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
        eye = np.eye(self.nbas)
        eris_ao = []
        for ci in self.clusters:
            Ci = self.Cact[:, ci]
            eris_ao.append(self.ao2mo((eye, Ci, Ci, Ci)))
        return eris_ao

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

    def cmf_ci(self, eris=None) -> float:
        """
        Converge the cluster states for the current orbitals (CMF-CI)

        Each cluster is solved with FCI in the mean field (J - K/2, built in the AO basis) of the core
        and the other clusters, until the cluster 1RDMs stop changing.

        Parameters
        ----------
        eris : list
            (tu|vw) of each cluster, computed if not given

        Returns
        -------
        e_tot : float
            CMF energy
        """
        log = logger.new_logger(self)
        nclusters = len(self.clusters)
        if self.cluster_1rdm is None:
            self.cluster_1rdm = self.get_init_guess_rdm1()

        hcore = self.mf.get_hcore()
        if eris is None:
            eris = self.get_cluster_eris()
        dm_core = self.get_core_dm()
        dms = self.get_cluster_dms()
        vcore, vclusters = self.get_veff()

        e_tot = 0.0
        conv = False
        for it in range(self.max_cycle_rdm1):
            d_err = 0.0
            for i in range(nclusters):
                dm_env = dm_core + sum(dms) - dms[i]
                v_env  = vcore + sum(vclusters) - vclusters[i]
                h0, h1 = self.get_cluster_hamiltonian(i, dm_env, v_env, hcore)
                e, ci, d1, d2 = self.solve_cluster(i, h0, h1, eris[i])

                d_err = max(d_err, np.linalg.norm(d1 - self.cluster_1rdm[i]))
                self.ci[i] = ci
                self.cluster_energies[i] = e
                self.cluster_1rdm[i] = d1
                self.cluster_2rdm[i] = d2

                if self.sequential:
                    Ci = self.Cact[:, self.clusters[i]]
                    dms[i] = Ci @ d1 @ Ci.T
                    vj, vk = self.mf.get_jk(self.mol, dms[i], hermi=1)
                    vclusters[i] = vj - .5*vk

            if not self.sequential and nclusters > 0:
                dms = self.get_cluster_dms()
                vj, vk = self.mf.get_jk(self.mol, np.asarray(dms), hermi=1)
                vclusters = list(vj - .5*vk)

            e_last, e_tot = e_tot, self.energy((vcore, vclusters), eris, hcore)
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

        self._veff = (vcore, vclusters)
        self._eris = eris
        self.e_tot = e_tot
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

    def get_gen_fock(self, veff=None) -> np.ndarray:
        """Generalized Fock matrix, F = 1/2 C^T dE/dC, in the [doc, act, vir] orbital basis
        """
        return .5 * self.get_mo_coeffs().T @ self.energy_deriv(veff)

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
        in the [doc, act, vir] orbital basis with the redundant rotations set to zero
        """
        F = self.get_gen_fock(veff)
        mask = self.rotation_mask()
        return 2*(F - F.T) * (mask | mask.T)

    def approx_hessian_diag(self, veff=None) -> np.ndarray:
        """
        Diagonal orbital Hessian guess, 2 (n_p f_qq + n_q f_pp - F_pp - F_qq), where f is the
        mean-field Fock matrix and F the generalized Fock matrix
        """
        if veff is None:
            veff = self.get_veff()
        vcore, vclusters = veff
        C = self.get_mo_coeffs()
        f = np.diag(C.T @ (self.mf.get_hcore() + vcore + sum(vclusters)) @ C)
        n = np.diag(self.get_occ())
        F = np.diag(self.get_gen_fock(veff))
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

        def energy_and_grad(x):
            K = unpack(x)
            self.set_mo_coeffs(C0 @ scipy.linalg.expm(K))
            eris_ao = self.get_cluster_eris_ao()
            e = self.cmf_ci(self.get_cluster_eris(eris_ao))
            # chain rule through exp(K): the adjoint of the Frechet derivative of exp at K is the one at K^T
            dEdU = C0.T @ self.energy_deriv(self._veff, eris_ao)
            dEdK = scipy.linalg.expm_frechet(K.T, dEdU, compute_expm=False)
            return e, (dEdK - dEdK.T)[mask]

        x = np.zeros(np.count_nonzero(mask))
        e, g = energy_and_grad(x)
        hdiag = np.maximum(abs(self.approx_hessian_diag(self._veff)[mask]), 0.1)
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
        log.note('CMF energy = %.12f', self.e_tot)
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
