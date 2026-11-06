"""
Active-space selection with cMF for [Cr2(OH)3(tmtacn)2]3+ (Pantazis geometry, def2-SVP):

    big       ooCMF in a generous AO-cluster space: Cr 3d4d4s4p (3 electrons each, antiparallel M_s) and the
              2s2p3s3p of the three bridging O (closed shell): (30e, 52o). The orbitals of the doubly occupied and
              virtual spaces are optimized with the active ones (trust-region Newton).
    select    natural occupations and single-orbital entropies of each cluster (cMF: the correlation within each
              cluster, exactly); keep the orbitals with min(n, 2 - n) >= --occ-thresh, fold the others into the
              doubly occupied / virtual spaces
    reduced   ooCMF in the selected active space; J from QDPT2 of the Cr-Cr pair (compare with the (24e, 38o) space
              of cr2_pantazis_ao.py, J = -16.45 cm-1); natural occupations of the cluster blocks of the PT1 1RDM of
              the Cr-Cr pair, which add the correlation between the two Cr: orbitals that it pushes across
              --occ-thresh show where the cMF-only selection is too tight or too loose

With --big-pt1 the PT1 1RDM of the Cr-Cr pair is computed in the big space and used for the selection, with J of the big
space. That needs QDPT2 of the 14-orbital Cr clusters, whose largest Fock sectors (3e+2e, 33124 determinants) are too
large to diagonalize: QDPT2 then falls back to the iterative 'cmf' partition in the 1e7-determinant space of the pair,
hours on 4 cores. The Cr-O pairs (22 orbitals, 1e9 determinants) are beyond the determinant-space QDPT2 either way.

    python cr2_active_space_selection.py --rohf-chk /path/to/rohf.chk     # rohf.chk of cr2_pantazis_ao.py

Results are appended to selection.json, the ooCMF iterations to pycmf.log; the orbitals of each ooCMF are saved after every iteration
(cmf_big.npy, cmf_reduced.npy) and the PT1 1RDM to pt1_dm1_big.npz, so the steps resume. J is for H = -2J S_A.S_B.
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
p.add_argument('--big-pt1', action='store_true', help='PT1 1RDM and J of the big space (slow, see above)')
p.add_argument('--tps-pt1', default=None,
               help='cluster-block RDMs of the big space from tps_pt1_cluster_rdms.jl (TPSPT1_NROOTS=4): natural orbitals '
                    'of --nat-root, and orbitals that carry the difference of --diff-roots are kept (--imp-thresh)')
p.add_argument('--nat-root', type=int, default=4, help='root of the reference space for the natural orbitals (S=0: 4)')
p.add_argument('--diff-roots', type=int, nargs=2, default=[4, 1], help='two roots (S=0, S=3: 4 1)')
p.add_argument('--imp-thresh', type=float, default=1e-6)
p.add_argument('--skip-big', action='store_true',
               help='cmf_big.npy is converged: only solve the cluster states (no orbital optimization, less memory)')
p.add_argument('--algorithm', default='newton', choices=['newton', 'lbfgs'],
               help='ooCMF optimizer (L-BFGS needs much less memory than the Newton caches)')
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
cmf.algorithm = args.algorithm
cmf.conv_tol_grad = args.conv_tol_grad
cmf.verbose, cmf.stdout = 4, open('pycmf.log', 'a')        # ooCMF iterations
cmf.chkfile = 'cmf_big.npy'
if os.path.exists('cmf_big.npy'):
    cmf.set_mo_coeffs(np.load('cmf_big.npy'))
    log('orbitals from cmf_big.npy')
log('big active space: (%de, %do), clusters %s, %d doubly occupied, %d virtual' % (
    sum(map(sum, fspace)), cmf.Cact.shape[1], [len(c) for c in cmf.clusters], cmf.Cdoc.shape[1], cmf.Cvir.shape[1]))

# ---------------------------------------------------------------- big
if args.skip_big:
    e_big = cmf.cmf_ci()
    cmf.e_tot, cmf.converged = e_big, True
else:
    e_big = cmf.kernel()
log('big ooCMF E = %.10f  converged %s' % (e_big, cmf.converged))
res['e_big'] = e_big

# ---------------------------------------------------------------- optional: PT1 between the Cr and J of the big space
dm1 = None
if args.big_pt1:
    if os.path.exists('pt1_dm1_big.npz'):                      # checkpoint
        d = np.load('pt1_dm1_big.npz')
        dm1, e2 = d['dm1'], float(d['e2'])
        log('PT1 1RDM from pt1_dm1_big.npz')
    else:
        e2, inc, dm1, _ = cmf.pt2_mbe(subsets=[(0, 1)], with_rdm=True)
        np.savez('pt1_dm1_big.npz', dm1=dm1, e2=e2)
    log('PT1 (Cr-Cr pair): E = %.10f' % e2)
    res['e_pt2_crcr_big'] = e2
    j, r = j_pair(cmf)
    log('big space: J = %.2f cm-1 (partition %s), levels (cm-1) %s' % (
        j, r.partition, ' '.join('S=%.0f:%.1f' % (s, (e - min(r.energies)) * CM) for e, s in sorted(zip(r.energies, r.spins)))))
    res['j_big'] = j

# ---------------------------------------------------------------- select
if args.tps_pt1:
    from pycmf.active_space import load_cluster_rdms
    rdms = load_cluster_rdms(args.tps_pt1, root=args.nat_root)
    diff = tuple(load_cluster_rdms(args.tps_pt1, root=r) for r in args.diff_roots)
    sel = cmf.select_active_space(occ_thresh=args.occ_thresh, rdms=rdms, diff_rdms=diff, imp_thresh=args.imp_thresh,
                                  verbose=4)
else:
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
if os.path.exists('cmf_reduced.npy') and np.load('cmf_reduced.npy').shape == cmf.get_mo_coeffs().shape:
    cmf.set_mo_coeffs(np.load('cmf_reduced.npy'))
    log('reduced orbitals from cmf_reduced.npy')
e_red = cmf.kernel()
log('reduced ooCMF E = %.10f  converged %s  (big - reduced: %.2e Eh)' % (e_red, cmf.converged, e_big - e_red))
res['e_reduced'] = e_red
j, r = j_pair(cmf)
log('reduced space: J = %.2f cm-1 (partition %s), levels (cm-1) %s' % (
    j, r.partition, ' '.join('S=%.0f:%.1f' % (s, (e - min(r.energies)) * CM) for e, s in sorted(zip(r.energies, r.spins)))))
res['j_reduced'], res['j_reduced_partition'] = j, r.partition

# ---------------------------------------------------------------- PT1 occupations of the selected space (diagnostic)
e2, inc, dm1, _ = cmf.pt2_mbe(subsets=[(0, 1)], with_rdm=True)
log('PT1 (Cr-Cr pair) in the reduced space: E = %.10f' % e2)
ent = cmf.orbital_entropies()
res['reduced'] = []
for i, ci in enumerate(cmf.clusters):
    occ = np.sort(np.linalg.eigvalsh(dm1[np.ix_(ci, ci)]))[::-1]
    near = [x for x in occ if min(x, 2 - x) < 2 * args.occ_thresh]
    log('cluster %d %s: cMF occupations %s; PT1 %s%s' % (
        i, tuple(cmf.fspace[i]), np.round(ent[i]['occ'], 4).tolist(), np.round(occ, 4).tolist(),
        ('  (%d within 2x the threshold)' % len(near)) if near else ''))
    res['reduced'].append({'occ_cmf': ent[i]['occ'].tolist(), 'occ_pt1': occ.tolist()})
old = json.load(open('selection.json')) if os.path.exists('selection.json') else []
json.dump(old + [res], open('selection.json', 'w'), indent=1, default=float)
