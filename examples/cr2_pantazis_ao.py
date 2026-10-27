"""
[Cr2(OH)3(tmtacn)2]3+ (Pantazis) with AO-based clusters and full orbital optimization:

    cmf     orbital-optimized RO-cMF (all doubly occupied, active and virtual orbitals optimized)
    qdpt2   spin ladder and exchange coupling J of the two Cr(III) (S = 3/2), H = -2J S_A.S_B
    pdft    cMF-PT1-PDFT: PDFT (tPBE, ftPBE) on the QDPT2 first-order wave function of each spin state, and J
    nevpt2  cMF-NEVPT2 correlation energy of the environment (frozen core), for the total energy

Default active space, as the 38-orbital TPSCI space of github.com/arnab82/cMF_data: Cr 3d+4d (10 orbitals, three
electrons, antiparallel M_s) and the 2p+3p of the three bridging O (6 orbitals, closed shell): (24e, 38o). The
clusters are meta-Lowdin orthogonalized AOs (orthonormal, any shells can be added with --cr-shells/--o-shells); the
orbital optimization then moves the bonding character that the doubly occupied orbitals lack at the start (the first
cMF energy is a few Eh above the ROHF). --guess spade starts from ROHF orbitals projected on the AOs instead.

    python cr2_pantazis_ao.py --steps scf,cmf,qdpt2,pdft,nevpt2
    python cr2_pantazis_ao.py --steps qdpt2,pdft --bridges          # reuses rohf.chk and cmf_orbitals.npy

Each step reads the checkpoints of the previous ones from the working directory (rohf.chk, cmf_orbitals.npy), so
steps can run as separate jobs; results are appended to results.json. Set OMP_NUM_THREADS to the cores of the node.

Notes
* QDPT2 and PT1-PDFT work in the determinant space of the clusters treated together. The Cr-Cr pair (20 orbitals,
  1.3e6 determinants) is fine; the bridging O enter its embedding as cMF mean field. With --bridges, the Cr-O-Cr
  triples are added (RDMs expanded over the pair and the triples, E_ot evaluated once per spin state, see
  pdft.mbe_spin_ladder), when their determinant space is at most --max-det: with the default space a triple has
  26 orbitals (5e10 determinants) and is skipped; with --cr-shells 3d --o-shells 2p it has 13 orbitals.
* cMF-NEVPT2 corrects the total energy of the RO-cMF reference, which averages over the spin ladder, so it does not
  change J. Order 1 (single clusters) is cheap; order 2 adds the pairs of clusters, whose 3- and 4-body reference RDMs
  in 12-20 orbitals take hours each.
"""
import argparse, json, os, time
import numpy as np
from pyscf import gto, scf, lib
from pyscf.data.elements import chemcore
from pyscf.fci import cistring
import pycmf
from pycmf import pdft

CM = 219474.6313705
here = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--steps', default='scf,cmf,qdpt2,pdft,nevpt2')
p.add_argument('--basis', default='def2-svp')
p.add_argument('--auxbasis', default='def2-universal-jkfit')
p.add_argument('--cr-shells', nargs='+', default=['3d', '4d'])
p.add_argument('--o-shells', nargs='+', default=['2p', '3p'])
p.add_argument('--geometry', default=os.path.join(here, 'data', 'cr2_pantazis.xyz'))
p.add_argument('--max-memory', type=int, default=16000, help='MB')
p.add_argument('--conv-tol-grad', type=float, default=1e-5, help='orbital gradient of ooCMF')
p.add_argument('--guess', default='ao', choices=['ao', 'spade'],
               help='cluster orbitals: the orthogonalized AOs themselves (ao), or the ROHF orbitals projected on them '
                    '(spade, a start closer to the optimized cMF)')
p.add_argument('--grids-level', type=int, default=3)
p.add_argument('--bridges', action='store_true', help='add the Cr-O-Cr triples to QDPT2 and PDFT')
p.add_argument('--max-det', type=float, default=2e7, help='largest determinant space of a QDPT2 subset')
p.add_argument('--nevpt2-order', type=int, default=1)
p.add_argument('--nevpt2-cutoff', type=float, default=None, help='Angstrom, for --nevpt2-order 2')
args = p.parse_args()
steps = args.steps.split(',')
results = {'args': vars(args)}
t0 = time.time()

def log(*a):
    print('[%6.0f s]' % (time.time() - t0), *a, flush=True)

def save():
    old = json.load(open('results.json')) if os.path.exists('results.json') else []
    json.dump(old + [results], open('results.json', 'w'), indent=1, default=float)

# ---------------------------------------------------------------- SCF (high-spin ROHF, S = 3)
mol = gto.M(atom=args.geometry, basis=args.basis, charge=3, spin=6, max_memory=args.max_memory, verbose=4,
            output='pycmf.log')                                  # SCF and ooCMF iterations
mf = scf.ROHF(mol).density_fit(auxbasis=args.auxbasis)
mf.chkfile = 'rohf.chk'
if os.path.exists('rohf.chk'):
    mf.__dict__.update(lib.chkfile.load('rohf.chk', 'scf'))
    log('ROHF %.10f (rohf.chk)' % mf.e_tot)
elif 'scf' in steps:
    mf.conv_tol = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    log('ROHF %.10f  converged %s' % (mf.e_tot, mf.converged))
else:
    raise RuntimeError('no rohf.chk: run the scf step first')
results['e_rohf'] = mf.e_tot
mol.verbose = mf.verbose = 0

# ---------------------------------------------------------------- clusters from AOs
labels = [['0 Cr %s' % s for s in args.cr_shells], ['1 Cr %s' % s for s in args.cr_shells]]
labels += [['%d O %s' % (ia, s) for s in args.o_shells] for ia in (2, 3, 4)]
fspace = [(3, 0), (0, 3), (3, 3), (3, 3), (3, 3)]
cmf = pycmf.CMF(mf)
cmf.set_clusters_from_ao(labels, fspace, guess=args.guess)
nact = cmf.Cact.shape[1]
log('active space: (%de, %do), clusters %s, %d doubly occupied, %d virtual' % (
    sum(map(sum, fspace)), nact, [len(c) for c in cmf.clusters], cmf.Cdoc.shape[1], cmf.Cvir.shape[1]))
cmf.conv_tol_grad = args.conv_tol_grad
cmf.verbose = 4
if 'cmf' in steps:
    if os.path.exists('cmf_orbitals.npy'):
        cmf.set_mo_coeffs(np.load('cmf_orbitals.npy'))
        log('restarting the orbital optimization from cmf_orbitals.npy')
    cmf.kernel()
    np.save('cmf_orbitals.npy', cmf.get_mo_coeffs())
elif os.path.exists('cmf_orbitals.npy'):
    cmf.set_mo_coeffs(np.load('cmf_orbitals.npy'))
    cmf.cmf_ci()
else:
    raise RuntimeError('no cmf_orbitals.npy: run the cmf step first')
cmf.verbose = 0
results['e_cmf'] = cmf.e_tot
results['cmf_converged'] = bool(cmf.converged)
log('cMF %.10f  converged %s' % (cmf.e_tot, cmf.converged))
save()

# ---------------------------------------------------------------- QDPT2 and PT1-PDFT of the spin ladder
def ndet(sub):
    n = sum(len(cmf.clusters[i]) for i in sub)
    na, nb = sum(fspace[i][0] for i in sub), sum(fspace[i][1] for i in sub)
    return cistring.num_strings(n, na) * cistring.num_strings(n, nb)

subsets = [(0, 1)]
if args.bridges:
    for k in (2, 3, 4):
        if ndet((0, 1, k)) <= args.max_det:
            subsets.append((0, 1, k))
        else:
            log('Cr-O-Cr triple (0, 1, %d): %.1e determinants > --max-det, skipped' % (k, ndet((0, 1, k))))

if 'qdpt2' in steps:
    js = {}
    for sub in subsets:
        res = cmf.qdpt2(list(sub))
        js[sub] = res.heisenberg_j()
        log('QDPT2 clusters %s (%.1e determinants): J = %.2f cm-1, levels %s' % (
            sub, ndet(sub), js[sub]*CM, ' '.join('S=%.0f:%.8f' % (s, e) for e, s in sorted(zip(res.energies, res.spins)))))
    j = js[(0, 1)] + sum(js[t] - js[(0, 1)] for t in subsets[1:])
    results['j_qdpt2_pair'] = js[(0, 1)]*CM
    results['j_qdpt2'] = j*CM
    log('QDPT2 J = %.2f cm-1  (%s)' % (j*CM, 'pair + Cr-O-Cr triples' if len(subsets) > 1 else 'Cr-Cr pair'))
    save()

if 'pdft' in steps:
    ots = {x: pdft.make_ot(mol, x, grids_level=args.grids_level) for x in ('tPBE', 'ftPBE')}
    for x, ot in ots.items():
        results['j_pdft_spin_coupled_' + x] = pdft.spin_ladder(cmf, ot=ot)[1]*CM
        log('PDFT on the spin-coupled cMF states (no charge transfer), %s: J = %.2f cm-1' % (x, results['j_pdft_spin_coupled_' + x]))
    rdms = pdft.mbe_rdm12(cmf, subsets)
    for x, ot in ots.items():
        energies, j = pdft.mbe_spin_ladder(cmf, ot=ot, rdms=rdms)
        results['j_pt1_pdft_' + x] = j*CM
        results['e_pt1_pdft_' + x] = {str(s): e for s, e in energies.items()}
        log('cMF-PT1-PDFT %s: J = %.2f cm-1, E(S) %s' % (x, j*CM, ' '.join('S=%.0f:%.8f' % kv for kv in energies.items())))
    save()

# ---------------------------------------------------------------- NEVPT2 (frozen core)
if 'nevpt2' in steps:
    from pycmf import mrpt
    ncore = chemcore(mol)
    fock = mrpt._prepare(cmf)['fock']
    e, u = np.linalg.eigh(cmf.Cdoc.T @ fock @ cmf.Cdoc)
    C = cmf.Cdoc @ u
    cmf.set_Cfrz(C[:, :ncore])
    cmf.set_Cdoc(C[:, ncore:])
    log('NEVPT2: %d frozen core orbitals, %d correlated doubly occupied' % (ncore, cmf.Cdoc.shape[1]))
    e_nev, inc = cmf.nevpt2_mbe(order=args.nevpt2_order, cutoff=args.nevpt2_cutoff)
    results['e_nevpt2'] = e_nev
    results['nevpt2_increments'] = {str(k): v for k, v in inc.items()}
    log('cMF-NEVPT2 (order %d): %.8f  E(cMF) + NEVPT2 = %.8f' % (args.nevpt2_order, e_nev, cmf.e_tot + e_nev))
    save()
log('done')
