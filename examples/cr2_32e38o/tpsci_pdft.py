"""
TPSCI-PDFT for the (32e, 38o) Cr2 problem: on-top functionals on the RDMs that tpsci_m.jl saved, for each threshold
and M, with J of H = -2J S_A.S_B (experiment -66 cm-1).

    python tpsci_pdft.py svp svp/M*/rdms_t*.npz
    python tpsci_pdft.py svp svp/M100/rdms_t04.npz --functionals tPBE,ftPBE,MC23

The first argument is the directory of make_integrals.py (orbitals.npz and rohf.chk, or --rohf-chk). Each RDM file
gets <file>_pdft.json (energies of each spin state and J for TPSCI, TPSCI+PT2, the RDM energy with density-fitted
integrals, and every functional); all files together go to <dir>/pdft_summary.csv, one row per RDM file.
"""
import argparse, csv, json, os, time
import numpy as np
from pyscf import gto, scf, lib
import pycmf
from pycmf import pdft

CM = 219474.6313705
FUNCTIONALS = 'tLDA,ftLDA,tPBE,ftPBE,tBLYP,ftBLYP,trevPBE,ftrevPBE,tOPBE,tM06L,tPBE0,MC23'
p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('dir', help='output directory of make_integrals.py')
p.add_argument('rdms', nargs='+', help='rdms_tNN.npz files of tpsci_m.jl')
p.add_argument('--functionals', default=FUNCTIONALS)
p.add_argument('--grids-level', type=int, default=3)
p.add_argument('--rohf-chk', default=None, help='default: <dir>/rohf.chk')
p.add_argument('--max-memory', type=int, default=10000, help='MB')
args = p.parse_args()
t0 = time.time()


def log(*a):
    print('[%6.0f s]' % (time.time() - t0), *a, flush=True)


def fit_j(levels):
    """J (cm-1) from E(S) = E0 - J S(S+1), levels {S: E}"""
    s = np.array(sorted(levels))
    A = np.vstack((np.ones_like(s), -s * (s + 1))).T
    return np.linalg.lstsq(A, np.array([levels[x] for x in s]), rcond=None)[0][1] * CM


# ---------------------------------------------------------------- the cMF orbitals of make_integrals.py
orb = np.load(os.path.join(args.dir, 'orbitals.npz'))
mol = gto.M(atom=str(orb['geometry']), basis=str(orb['basis']), charge=3, spin=6, max_memory=args.max_memory, verbose=0)
mf = scf.ROHF(mol).density_fit(auxbasis=str(orb['auxbasis']))
chk = args.rohf_chk or os.path.join(args.dir, 'rohf.chk')
mf.__dict__.update(lib.chkfile.load(chk, 'scf'))
cl = orb['clusters']
clusters = [list(np.where(cl == i)[0]) for i in range(cl.max() + 1)]
fspace = [tuple(int(x) for x in f) for f in orb['fspace']]
cmf = pycmf.CMF(mf)
cmf.set_Cdoc(orb['Cdoc'])
cmf.set_Cact(orb['Cact'], clusters, fspace)
cmf.set_Cvir(orb['Cvir'])
h0_a, h1_a, h2_a, orbs = cmf.get_active_hamiltonian()
assert list(orbs) == list(range(orb['Cact'].shape[1])), 'active orbitals must be in cluster order'
functionals = args.functionals.split(',')
log('%s, (%de, %do), %d functionals' % (orb['basis'], sum(map(sum, fspace)), len(orbs), len(functionals)))

rows = []
for fn in args.rdms:
    d = np.load(fn)
    e, dm1s, dm2s = pdft.load_tpsci_rdms(fn)
    e2 = np.asarray(d['e2a']).ravel() if 'e2a' in d.files else np.zeros_like(e)
    s2 = np.asarray(d['s2']).ravel() if 's2' in d.files else [pdft.spin_square_rdm(a, b) for a, b in zip(dm1s, dm2s)]
    rdms, lev_var, lev_pt2, lev_rdm = {}, {}, {}, {}
    for r in range(len(e)):
        s = round(np.sqrt(1 + 4 * max(s2[r], 0)) - 1) / 2
        e_rdm = h0_a + np.einsum('pq,pq', h1_a, dm1s[r]) + .5 * np.einsum('pqrs,pqrs', h2_a, dm2s[r])
        if s in rdms:
            continue                                   # lowest root of each spin
        rdms[s] = (dm1s[r], dm2s[r])
        lev_var[s], lev_pt2[s], lev_rdm[s] = e[r], e[r] + e2[r], e_rdm
    out = {'file': fn, 'M': int(np.ravel(d['M'])[0]) if 'M' in d.files else None,
           'thresh': float(np.ravel(d['thresh'])[0]) if 'thresh' in d.files else None,
           'ncfg': int(np.ravel(d['ncfg'])[0]) if 'ncfg' in d.files else None,
           'spins': sorted(rdms), 'j_tpsci_var': fit_j(lev_var), 'j_tpsci_pt2': fit_j(lev_pt2),
           'j_rdm_df': fit_j(lev_rdm), 'e_tpsci': {str(k): v for k, v in lev_var.items()}}
    scan = pdft.functional_scan(cmf, rdms, functionals, grids_level=args.grids_level)
    for x in functionals:
        out['j_pdft_' + x] = scan[x][1] * CM
        out['e_pdft_' + x] = {str(k): v for k, v in scan[x][0].items()}
    json.dump(out, open(os.path.splitext(fn)[0] + '_pdft.json', 'w'), indent=1, default=float)
    log('%s: M %s thresh %s ncfg %s  J(TPSCI) %.2f  J(+PT2) %.2f  J(PDFT) %s' % (
        fn, out['M'], out['thresh'], out['ncfg'], out['j_tpsci_var'], out['j_tpsci_pt2'],
        ' '.join('%s %.2f' % (x, out['j_pdft_' + x]) for x in functionals)))
    rows.append(out)

cols = ['file', 'M', 'thresh', 'ncfg', 'j_tpsci_var', 'j_tpsci_pt2', 'j_rdm_df'] + ['j_pdft_' + x for x in functionals]
summary = os.path.join(args.dir, 'pdft_summary.csv')
new = not os.path.exists(summary)
with open(summary, 'a', newline='') as f:
    w = csv.writer(f)
    if new:
        w.writerow(cols)
    for r in rows:
        w.writerow([r[c] if not isinstance(r[c], float) else '%.4f' % r[c] for c in cols])
log('appended %d rows to %s' % (len(rows), summary))
