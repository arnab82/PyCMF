"""
Diabatic states from the character of cluster-state wavefunctions (TPSCI, with one cluster per chromophore): every
configuration belongs to a class (GS, LE_i, CT_i_j, TT_i_j, ...), and W[k][I, J] = sum over the configurations of
class k of c_I c_J gives the weight of class k in each model state and their overlaps. The diabat of class k is the
model-space vector with the largest weight in k (top eigenvector of W[k]); the diabats are Lowdin orthonormalized
(closest orthonormal set) and any Hamiltonian on the model space (wavefunction, L-PDFT) is transformed to them.

For singlet fission: the direct coupling LE_i - TT and the CT-mediated (superexchange) one,
V_eff = sum_CT V(LE, CT) V(CT, TT) * 1/2 [1/(E_LE - E_CT) + 1/(E_TT - E_CT)].
"""
import numpy as np


def diabatize(W, names, H, exclude=('other',), nstates=None):
    """
    Parameters
    ----------
    W : (n, n, K) class weight matrices of the n model states (e.g. the singlets), K classes
    names : list of the K class names
    H : (n, n) Hamiltonian of the model states (diag(E) for eigenstates, or the L-PDFT Hamiltonian)
    exclude : classes not used as diabats
    nstates : number of diabats (default n): the classes with the largest top weights

    Returns
    -------
    dnames : names of the diabats
    D : (n, m) orthonormal diabats in the basis of the model states (columns)
    Hd : (m, m) diabatic Hamiltonian
    weight : (m,) class weight of each diabat before orthonormalization
    """
    n = H.shape[0]
    m = n if nstates is None else nstates
    cand = []
    for k, name in enumerate(names):
        if name in exclude:
            continue
        w, v = np.linalg.eigh(W[:, :, k])
        x = v[:, -1]
        x = x * np.sign(x[np.argmax(abs(x))])
        cand.append((w[-1], name, x))
    cand.sort(key=lambda t: -t[0])
    cand = cand[:m]
    dnames = [c[1] for c in cand]
    D = np.array([c[2] for c in cand]).T
    S = D.T @ D
    e, u = np.linalg.eigh(S)
    if e.min() < 1e-6:
        raise ValueError('diabats of classes %s are linearly dependent in the model space' % dnames)
    D = D @ (u @ np.diag(e**-.5) @ u.T)
    return dnames, D, D.T @ H @ D, np.array([c[0] for c in cand])


def sf_couplings(dnames, Hd):
    """
    Direct and CT-mediated couplings between each LE diabat and each TT diabat

    Returns
    -------
    list of dict: le, tt, direct V(LE, TT), mediated V_eff (superexchange over all CT diabats), and the terms
    """
    idx = {k: i for i, k in enumerate(dnames)}
    E = np.diag(Hd)
    out = []
    for le in [k for k in dnames if k.startswith('LE')]:
        for tt in [k for k in dnames if k.startswith('TT')]:
            i, j = idx[le], idx[tt]
            terms = {}
            for ct in [k for k in dnames if k.startswith('CT')]:
                c = idx[ct]
                terms[ct] = Hd[i, c] * Hd[c, j] * .5 * (1 / (E[i] - E[c]) + 1 / (E[j] - E[c]))
            out.append({'le': le, 'tt': tt, 'direct': Hd[i, j], 'mediated': sum(terms.values()), 'terms': terms})
    return out
