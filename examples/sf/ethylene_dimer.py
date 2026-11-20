"""
Singlet fission prototype: L-PDFT on TPSCI states of a stacked ethylene dimer, the minimal singlet-fission model
(each monomer's pi/pi* is a cluster; the low singlets are S0, the local excitations, the charge-transfer states and
the singlet triplet pair 1TT).

    python ethylene_dimer.py prepare --out ed          # cMF clusters, ooCMF, export (ed/cmf.npz)
    julia --project=/path/to/TPSChem.jl -t auto tpsci_states.jl ed/cmf.npz ed/states.npz
    python ethylene_dimer.py analyze --out ed          # L-PDFT on the TPSCI singlets, check against pyscf-forge

In this (4e, 4o) space TPSCI with tight thresholds is exact, so L-PDFT on its singlets must equal pyscf-forge's L-PDFT
on CASCI singlets in the same orbitals: the check of the whole chain (TPSCI RDMs and transition RDMs, conventions,
orbital order). Then, the character of each state in the cluster basis (LE / CT / TT) and the L-PDFT mixing.
"""
import argparse, json, os
import numpy as np
from pyscf import gto, scf, lib, mcpdft, fci
import pycmf
from pycmf.lpdft import lpdft_from_cmf
from pycmf.diabatic import diabatize, sf_couplings

EV = 27.211386245988

p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('step', choices=['prepare', 'analyze'])
p.add_argument('--out', default='ed')
p.add_argument('--distance', type=float, default=3.5, help='stacking distance (Angstrom)')
p.add_argument('--slip', type=float, default=0.0, help='slip of the second monomer along the C=C axis (Angstrom)')
p.add_argument('--basis', default='6-31g')
p.add_argument('--functionals', default='tPBE,ftPBE,tBLYP,tPBE0')
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

R, z = args.distance, args.slip
mol = gto.M(atom=f'''C 0 0 0.667; C 0 0 -0.667; H 0 0.923 1.237; H 0 -0.923 1.237; H 0 0.923 -1.237; H 0 -0.923 -1.237;
C {R} 0 {0.667+z}; C {R} 0 {-0.667+z}; H {R} 0.923 {1.237+z}; H {R} -0.923 {1.237+z}; H {R} 0.923 {-1.237+z};
H {R} -0.923 {-1.237+z}''',
            basis=args.basis, verbose=0)
mf = scf.RHF(mol).density_fit()
chk = os.path.join(args.out, 'rhf.chk')
if os.path.exists(chk):
    mf.__dict__.update(lib.chkfile.load(chk, 'scf'))
else:
    mf.chkfile = chk
    mf.run(conv_tol=1e-11)
cmf = pycmf.CMF(mf)
# pi/pi* of each monomer: the C 2px (and 3px) AOs, monomers stacked along x
shells = ['2px', '3px'] if args.basis.lower() != 'sto-3g' else ['2px']
cmf.set_clusters_from_ao([['%d C %s' % (a, s) for a in (0, 1) for s in shells],
                          ['%d C %s' % (a, s) for a in (6, 7) for s in shells]], [(1, 1), (1, 1)])
cmf.algorithm = 'newton'
orbfile = os.path.join(args.out, 'cmf_orbitals.npy')

if args.step == 'prepare':
    e = cmf.kernel()
    np.save(orbfile, cmf.get_mo_coeffs())
    cmf.export_fermicg(os.path.join(args.out, 'cmf.npz'))
    print('RHF %.10f  ooCMF %.10f  clusters %s -> %s/cmf.npz' % (mf.e_tot, e, [len(c) for c in cmf.clusters], args.out))
    raise SystemExit

cmf.set_mo_coeffs(np.load(orbfile))
cmf.cmf_ci()
d = np.load(os.path.join(args.out, 'states.npz'))
e, s2, pairs = d['energies'], d['s2'], d['pairs']
singlets = [r for r in range(len(e)) if s2[r] < 0.1]
ns = len(singlets)
print('TPSCI roots: E %s\n             <S^2> %s\nsinglets: %s' % (np.round(e, 6), np.round(s2, 3), singlets))
chan = {(int(a) - 1, int(b) - 1): k for k, (a, b) in enumerate(pairs)}


def to_pyscf(G):
    """TPSChem G[p,q,r,s] = <p'q'sr> -> PySCF dm2[p,q,r,s] = <p'r'sq>"""
    return G.transpose(0, 2, 1, 3)


n = cmf.Cact.shape[1]
dm1 = np.array([d['dm1'][:, :, chan[r, r]] for r in singlets])
dm2 = np.array([to_pyscf(d['dm2'][:, :, :, :, chan[r, r]]) for r in singlets])
t1, t2 = {}, {}
for a in range(ns):
    for b in range(a + 1, ns):
        k = chan[singlets[a], singlets[b]]
        t1[a, b], t2[a, b] = d['dm1'][:, :, k], to_pyscf(d['dm2'][:, :, :, :, k])

out = {'tpsci_singlets': e[singlets].tolist()}
C = np.hstack((cmf.Cfrz, cmf.Cdoc, cmf.Cact, cmf.Cvir))
ncore = cmf.Cfrz.shape[1] + cmf.Cdoc.shape[1]
for otxc in args.functionals.split(','):
    H, eL, U = lpdft_from_cmf(cmf, dm1, dm2, t1, t2, otxc=otxc)
    # reference: pyscf-forge L-PDFT on CASCI singlets in the same orbitals
    ref = mcpdft.CASCI(mf, otxc, n, (2, 2))
    ref.verbose = 0
    ref.fcisolver = fci.direct_spin1.FCI(mol)
    ref.fcisolver.nroots = ns
    ref.fcisolver.conv_tol, ref.fcisolver.max_cycle = 1e-14, 500   # the default leaves CI residuals ~1e-5
    ref.fix_spin_(ss=0)
    ref.canonicalization = False
    lm = ref.multi_state([1 / ns] * ns, method='lin')
    lm.kernel(mo_coeff=C, dump_chk=False)
    print('%-6s L-PDFT(TPSCI) %s\n       L-PDFT(CASCI, pyscf-forge) %s   max diff %.1e' % (
        otxc, np.round(eL, 8), np.round(np.sort(lm.e_states), 8), abs(eL - np.sort(lm.e_states)).max()))
    print('       mixing of the TPSCI singlets in the L-PDFT states (|U|^2):\n%s' % np.round(U**2, 3))
    out[otxc] = {'e': eL.tolist(), 'e_forge': np.sort(lm.e_states).tolist(), 'H': H.tolist()}
# ---------------------------------------------------------------- diabatic LE / CT / TT analysis
names = open(os.path.join(args.out, 'states_classes.txt')).read().split()
Wsing = d['class_weights'][np.ix_(singlets, singlets)]


def report(label, Hmodel):
    dn, D, Hd, w = diabatize(Wsing, names, Hmodel)
    e0 = Hd[dn.index('GS'), dn.index('GS')] if 'GS' in dn else 0
    print('%s diabats (class weight before orthonormalization): %s' % (label, ', '.join('%s %.3f' % x for x in zip(dn, w))))
    print('   diabatic energies (eV above GS): %s' % ', '.join('%s %.3f' % (k, (Hd[i, i] - e0) * EV) for i, k in enumerate(dn)))
    for c in sf_couplings(dn, Hd):
        print('   %s -> %s: direct %8.2f meV, CT-mediated %8.2f meV (%s)' % (
            c['le'], c['tt'], c['direct'] * EV * 1e3, c['mediated'] * EV * 1e3,
            ', '.join('%s %.2f' % (k, v * EV * 1e3) for k, v in c['terms'].items())))
    return {'diabats': dn, 'Hd_eV': (Hd * EV).tolist(), 'couplings_meV': [
        {k: (v * EV * 1e3 if k in ('direct', 'mediated') else v) for k, v in c.items() if k != 'terms'}
        for c in sf_couplings(dn, Hd)]}


out['diabatic'] = {'wfn': report('TPSCI', np.diag(e[singlets]))}
for otxc in args.functionals.split(','):
    out['diabatic'][otxc] = report('L-PDFT ' + otxc, np.array(out[otxc]['H']))
json.dump(out, open(os.path.join(args.out, 'lpdft.json'), 'w'), indent=1)
