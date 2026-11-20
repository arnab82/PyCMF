"""
Tests for the class-based diabatization (pycmf.diabatic)
"""
import numpy as np
from pycmf.diabatic import diabatize, sf_couplings


def test_recovers_diabatic_hamiltonian():
    """Model states that are rotations of class configurations: the diabats are those configurations and the
    diabatic Hamiltonian is recovered up to the phases of the diabats"""
    rng = np.random.default_rng(1)
    names = ['GS', 'LE_1', 'CT_1_2', 'TT_1_2']
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    W = np.array([np.outer(Q[k], Q[k]) for k in range(4)]).transpose(1, 2, 0)
    Hd0 = rng.standard_normal((4, 4))
    Hd0 = Hd0 + Hd0.T
    dn, D, Hd, w = diabatize(W, names, Q.T @ Hd0 @ Q)
    perm = [names.index(x) for x in dn]
    assert np.allclose(w, 1)
    assert np.allclose(abs(Hd), abs(Hd0[np.ix_(perm, perm)]), atol=1e-12)
    assert np.allclose(D.T @ D, np.eye(4), atol=1e-12)


def test_superexchange():
    """Three-state LE / CT / TT model: direct and second-order CT-mediated coupling"""
    names = ['LE_1', 'CT_1_2', 'TT_1_2']
    Hd = np.array([[0.0, 0.10, 0.01], [0.10, 0.5, 0.05], [0.01, 0.05, -0.1]])
    W = np.array([np.diag(np.eye(3)[k]) for k in range(3)]).transpose(1, 2, 0)
    dn, D, Hd2, w = diabatize(W, names, Hd)
    c = sf_couplings(dn, Hd2)[0]
    assert abs(abs(c['direct']) - 0.01) < 1e-12
    assert abs(abs(c['mediated']) - abs(0.10 * 0.05 * .5 * (1 / (0 - .5) + 1 / (-.1 - .5)))) < 1e-12
