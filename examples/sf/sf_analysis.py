"""
Shared analysis of TPSCI states for singlet fission (used by ethylene_dimer.py and acene_dimer.py): L-PDFT on the TPSCI
singlets (optionally checked against pyscf-forge's L-PDFT on CASCI singlets in the same orbitals) and the diabatic
LE / CT / TT analysis with direct and CT-mediated couplings.
"""
import json, os
import numpy as np
from pyscf import mcpdft, fci
from pycmf.lpdft import lpdft_from_cmf
from pycmf.diabatic import diabatize, sf_couplings

EV = 27.211386245988


def to_pyscf(G):
    """TPSChem G[p,q,r,s] = <p'q'sr> -> PySCF dm2[p,q,r,s] = <p'r'sq>"""
    return G.transpose(0, 2, 1, 3)


def load_singlets(outdir):
    d = np.load(os.path.join(outdir, 'states.npz'))
    e, s2, pairs = d['energies'], d['s2'], d['pairs']
    singlets = [r for r in range(len(e)) if s2[r] < 0.1]
    chan = {(int(a) - 1, int(b) - 1): k for k, (a, b) in enumerate(pairs)}
    dm1 = np.array([d['dm1'][:, :, chan[r, r]] for r in singlets])
    dm2 = np.array([to_pyscf(d['dm2'][:, :, :, :, chan[r, r]]) for r in singlets])
    t1, t2 = {}, {}
    for a in range(len(singlets)):
        for b in range(a + 1, len(singlets)):
            k = chan[singlets[a], singlets[b]]
            t1[a, b], t2[a, b] = d['dm1'][:, :, k], to_pyscf(d['dm2'][:, :, :, :, k])
    names = open(os.path.join(outdir, 'states_classes.txt')).read().split()
    W = d['class_weights'][np.ix_(singlets, singlets)]
    print('TPSCI roots: E %s\n             <S^2> %s\nsinglets: %s' % (np.round(e, 6), np.round(s2, 3), singlets))
    return e[singlets], dm1, dm2, t1, t2, names, W


def check_casci(cmf, nelecas, ns, otxc, eL):
    """pyscf-forge L-PDFT on the lowest ns CASCI singlets in the orbitals of cmf"""
    mf, mol = cmf.mf, cmf.mol
    C = np.hstack((cmf.Cfrz, cmf.Cdoc, cmf.Cact, cmf.Cvir))
    ref = mcpdft.CASCI(mf, otxc, cmf.Cact.shape[1], nelecas)
    ref.verbose = 0
    ref.fcisolver = fci.direct_spin1.FCI(mol)
    ref.fcisolver.nroots = ns
    ref.fcisolver.conv_tol, ref.fcisolver.max_cycle = 1e-14, 500   # the default leaves CI residuals ~1e-5
    ref.fix_spin_(ss=0)
    ref.canonicalization = False
    lm = ref.multi_state([1 / ns] * ns, method='lin')
    lm.kernel(mo_coeff=C, dump_chk=False)
    ef = np.sort(lm.e_states)
    print('       L-PDFT(CASCI, pyscf-forge) %s   max diff %.1e' % (np.round(ef, 8), abs(eL - ef).max()))
    return ef


def report_diabatic(label, W, names, Hmodel):
    dn, D, Hd, w = diabatize(W, names, Hmodel)
    e0 = Hd[dn.index('GS'), dn.index('GS')] if 'GS' in dn else 0
    print('%s diabats (class weight): %s' % (label, ', '.join('%s %.3f' % x for x in zip(dn, w))))
    print('   diabatic energies (eV above GS): %s' % ', '.join('%s %.3f' % (k, (Hd[i, i] - e0) * EV)
                                                         for i, k in enumerate(dn)))
    cps = sf_couplings(dn, Hd)
    for c in cps:
        print('   %s -> %s: direct %8.2f meV, CT-mediated %8.2f meV (%s)' % (
            c['le'], c['tt'], c['direct'] * EV * 1e3, c['mediated'] * EV * 1e3,
            ', '.join('%s %.2f' % (k, v * EV * 1e3) for k, v in c['terms'].items())))
    return {'diabats': dn, 'weights': w.tolist(), 'Hd_eV': (Hd * EV).tolist(),
            'couplings_meV': [{'le': c['le'], 'tt': c['tt'], 'direct': c['direct'] * EV * 1e3,
                               'mediated': c['mediated'] * EV * 1e3} for c in cps]}


def analyze(cmf, outdir, nelecas, functionals, check=True):
    """L-PDFT and diabatic analysis of the TPSCI singlets in outdir/states.npz; results to outdir/lpdft.json"""
    e, dm1, dm2, t1, t2, names, W = load_singlets(outdir)
    ns = len(e)
    out = {'tpsci_singlets': e.tolist(), 'diabatic': {'wfn': report_diabatic('TPSCI', W, names, np.diag(e))}}
    for otxc in functionals:
        H, eL, U = lpdft_from_cmf(cmf, dm1, dm2, t1, t2, otxc=otxc)
        print('%-6s L-PDFT(TPSCI) %s' % (otxc, np.round(eL, 8)))
        out[otxc] = {'e': eL.tolist(), 'H': H.tolist()}
        if check:
            out[otxc]['e_forge'] = check_casci(cmf, nelecas, ns, otxc, eL).tolist()
        out['diabatic'][otxc] = report_diabatic('L-PDFT ' + otxc, W, names, H)
    json.dump(out, open(os.path.join(outdir, 'lpdft.json'), 'w'), indent=1)
    return out
