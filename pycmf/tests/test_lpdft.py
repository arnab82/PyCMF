"""
L-PDFT from RDMs (pycmf.lpdft) against pyscf-forge's L-PDFT on the same CASCI states
"""
import numpy as np
import pytest
from pyscf import gto, scf, mcpdft, fci

from pycmf.lpdft import lpdft_hamiltonian


@pytest.fixture(scope='module')
def ethylene_dimer():
    """Stacked ethylene dimer (3.5 A), CAS(4,4) of the pi/pi* of both monomers: the minimal singlet-fission model"""
    mol = gto.M(atom='''C 0 0 0.667; C 0 0 -0.667; H 0 0.923 1.237; H 0 -0.923 1.237; H 0 0.923 -1.237;
                        H 0 -0.923 -1.237; C 3.5 0 0.667; C 3.5 0 -0.667; H 3.5 0.923 1.237; H 3.5 -0.923 1.237;
                        H 3.5 0.923 -1.237; H 3.5 -0.923 -1.237''', basis='sto-3g', verbose=0)
    return scf.RHF(mol).density_fit().run(conv_tol=1e-11)


@pytest.mark.parametrize('otxc', ['tPBE', 'ftPBE', 'tPBE0'])
def test_lpdft_matches_pyscf_forge(ethylene_dimer, otxc):
    """Four singlets: the L-PDFT Hamiltonian from state and transition RDMs is diagonal in pyscf-forge's L-PDFT
    eigenstates, with its energies (the hybrid adds the wavefunction Hamiltonian, also from the RDMs)"""
    mf = ethylene_dimer
    R = 4
    ref = mcpdft.CASCI(mf, otxc, 4, 4)
    ref.verbose = 0
    ref.fcisolver = fci.direct_spin1.FCI(mf.mol)
    ref.fcisolver.nroots = R
    ref.fix_spin_(ss=0)
    lm = ref.multi_state([1 / R] * R, method='lin')
    lm.kernel(dump_chk=False)
    dm1, dm2 = np.zeros((R, 4, 4)), np.zeros((R, 4, 4, 4, 4))
    t1, t2 = {}, {}
    for i in range(R):
        dm1[i], dm2[i] = fci.direct_spin1.make_rdm12(lm.ci[i], 4, (2, 2))
        for j in range(i + 1, R):
            t1[i, j], t2[i, j] = fci.direct_spin1.trans_rdm12(lm.ci[i], lm.ci[j], 4, (2, 2))
    H, e, U = lpdft_hamiltonian(mf, lm.mo_coeff, ref.ncore, 4, (2, 2), dm1, dm2, t1, t2, otxc=otxc)
    assert np.allclose(H, np.diag(lm.e_states), atol=1e-10)
    assert np.allclose(e, np.sort(lm.e_states), atol=1e-10)
