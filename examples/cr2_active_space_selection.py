"""
Active-space selection with cMF for [Cr2(OH)3(tmtacn)2]3+ (Pantazis geometry, def2-SVP):

    big       ooCMF in a generous AO-cluster space: Cr 3d4d4s4p (3 electrons each, antiparallel M_s) and the
              2s2p3s3p of the three bridging O (closed shell): (30e, 52o). The orbitals of the doubly occupied and
              virtual spaces are optimized with the active ones (trust-region Newton).
    pt1       first-order correlation between the two Cr clusters (QDPT2 of the Cr-Cr pair, the clusters treated
              together), as the 1RDM of a many-body expansion that has only that pair (Cr-O pairs, 22 orbitals and
              1e9 determinants, are beyond the determinant-space QDPT2); also J of the big space
    select    natural occupations and single-orbital entropies of each cluster (cMF), and occupations of the cluster
              blocks of the PT1 1RDM; keep the orbitals with min(n, 2 - n) >= --occ-thresh, fold the others into
              the doubly occupied / virtual spaces
    reduced   ooCMF in the selected active space, and J from QDPT2 of the Cr-Cr pair, to compare with the big space
              and with the (24e, 38o) space of cr2_pantazis_ao.py (J = -16.45 cm-1 there)

    python cr2_active_space_selection.py --rohf-chk /path/to/rohf.chk     # rohf.chk of cr2_pantazis_ao.py

Results are appended to selection.json, the ooCMF iterations to pycmf.log; the orbitals of each ooCMF are saved after every iteration
(cmf_big.npy, cmf_reduced.npy), so the steps resume. J is for H = -2J S_A.S_B.
"""
import argparse, json, os, time
import numpy as np
from pyscf import gto, scf, lib
import pycmf

CM = 219474.6313705
here = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--rohf-chk', default='rohf.chk')
p.add_argument('--geometry', default=os.path.join(here, 'data', 'cr2_pantazis.xyz'))
p.add_argument('--basis', default='def2-svp')
p.add_argument('--auxbasis', default='def2-universal-jkfit')
p.add_argument('--cr-shells', nargs='+', default=['3d', '4d', '4s', '4p'])
p.add_argument('--o-shells', nargs='+', default=['2s', '2p', '3s', '3p'])
p.add_argument('--guess', default='spade', choices=['ao', 'spade'])
p.add_argument('--occ-thresh', type=float, default=0.02)
p.add_argument('--no-pt1', action='store_true', help='select from the cMF occupations only')
p.add_argument('--conv-tol-grad', type=float, default=1e-5)
p.add_argument('--max-memory', type=int, default=12000)
args = p.parse_args()
t0 = time.time()
res = {'args': vars(args)}


def log(*a):
    print('[%6.0f s]' % (time.time() - t0), *a, flush=True)


def j_pair(cmf):
    """J of the two Cr from QDPT2 of the Cr-Cr pair (clusters 0 and 1) in the mean field of the bridges"""
    r = cmf.qdpt2(clusters=[0, 1])
    return r.heisenberg_j() * CM, r


mol = gto.M(atom=args.geometry, basis=args.basis, charge=3, spin=6, max_memory=args.max_memory, verbose=0)
mf = scf.ROHF(mol).density_fit(auxbasis=args.auxbasis)
mf.__dict__.update(lib.chkfile.load(args.rohf_chk, 'scf'))
log('ROHF %.10f' % mf.e_tot)

labels = [['0 Cr %s' % s for s in args.cr_shells], ['1 Cr %s' % s for s in args.cr_shells]]
labels += [['%d O %s' % (ia, s) for s in args.o_shells] for ia in (2, 3, 4)]
# the bridging O (OH-) are closed shell: 2s2p hold 8 electrons; with only 2p (and 3p): 6
nel_o = (2 if any(s == '2s' for s in args.o_shells) else 0) + (6 if any(s == '2p' for s in args.o_shells) else 0)
fspace = [(3, 0), (0, 3)] + [(nel_o // 2, nel_o // 2)] * 3
cmf = pycmf.CMF(mf)
cmf.set_clusters_from_ao(labels, fspace, guess=args.guess)
cmf.algorithm = 'newton'
cmf.conv_tol_grad = args.conv_tol_grad
cmf.verbose, cmf.stdout = 4, open('pycmf.log', 'a')        # ooCMF iterations
cmf.chkfile = 'cmf_big.npy'
if os.path.exists('cmf_big.npy'):
    cmf.set_mo_coeffs(np.load('cmf_big.npy'))
    log('orbitals from cmf_big.npy')
log('big active space: (%de, %do), clusters %s, %d doubly occupied, %d virtual' % (
    sum(map(sum, fspace)), cmf.Cact.shape[1], [len(c) for c in cmf.clusters], cmf.Cdoc.shape[1], cmf.Cvir.shape[1]))

# ---------------------------------------------------------------- big
e_big = cmf.kernel()
log('big ooCMF E = %.10f  converged %s' % (e_big, cmf.converged))
res['e_big'] = e_big

# ---------------------------------------------------------------- PT1 between the Cr and J of the big space
dm1 = None
if not args.no_pt1:
    e2, inc, dm1, _ = cmf.pt2_mbe(subsets=[(0, 1)], with_rdm=True)
    log('PT1 (Cr-Cr pair): E = %.10f' % e2)
    res['e_pt2_crcr_big'] = e2
j, r = j_pair(cmf)
log('big space: J = %.2f cm-1, levels (cm-1) %s' % (j, ' '.join('S=%.0f:%.1f' % (s, (e - min(r.energies)) * CM)
                                                          for e, s in sorted(zip(r.energies, r.spins)))))
res['j_big'] = j

# ---------------------------------------------------------------- select
sel = cmf.select_active_space(occ_thresh=args.occ_thresh, dm1=dm1, verbose=4)
ent = cmf.orbital_entropies()
res['big'] = [{'occ_cmf': e['occ'].tolist(), 'entropy': e['entropy'].tolist(),
               'occ_selection': [r[0] for r in t], 'where': [r[2] for r in t]} for e, t in zip(ent, sel['table'])]
log('cMF entropies per cluster: %s' % [np.round(e['entropy'], 4).tolist() for e in ent])
cmf.reduce_active_space_(sel)
log('selected active space: (%de, %do), clusters %s, %d doubly occupied, %d virtual' % (
    sum(map(sum, cmf.fspace)), cmf.Cact.shape[1], [len(c) for c in cmf.clusters], cmf.Cdoc.shape[1],
    cmf.Cvir.shape[1]))
res['selected'] = {'fspace': cmf.fspace, 'clusters': [len(c) for c in cmf.clusters]}

# ---------------------------------------------------------------- reduced
cmf.chkfile = 'cmf_reduced.npy'
e_red = cmf.kernel()
log('reduced ooCMF E = %.10f  converged %s  (big - reduced: %.2e Eh)' % (e_red, cmf.converged, e_big - e_red))
res['e_reduced'] = e_red
j, r = j_pair(cmf)
log('reduced space: J = %.2f cm-1, levels (cm-1) %s' % (j, ' '.join('S=%.0f:%.1f' % (s, (e - min(r.energies)) * CM)
                                                              for e, s in sorted(zip(r.energies, r.spins)))))
res['j_reduced'] = j
old = json.load(open('selection.json')) if os.path.exists('selection.json') else []
json.dump(old + [res], open('selection.json', 'w'), indent=1, default=float)
