"""
(32e, 38o) active space of [Cr2(OH)3(tmtacn)2]3+ with AO-cMF, in any basis, for TPSCI (TPSChem.jl) and TPSCI-PDFT:

    clusters   Cr 3d4d (10 orbitals; 5 alpha + 2 beta electrons, the second Cr with antiparallel M_s: (2, 5)) and the
               2p3p of each bridging O (6 orbitals, (3, 3)): the space of the 2024 Faraday Discussions TPSCI data
    orbitals   ROHF orbitals projected on the cluster AOs (SPADE: SVD of the doubly occupied, singly occupied and
               virtual blocks, as the SVD partitioning of the Faraday recipe), then ooCMF (trust-region Newton), which
               optimizes the doubly occupied and virtual orbitals together with the active ones (the Faraday cmf.jl
               only rotated within the active space)

    python make_integrals.py --basis def2-svp  --out svp
    python make_integrals.py --basis def2-tzvp --out tzvp --guess-chk svp/rohf.chk   # ~1200 AOs: >= 32 GB

Writes, in --out:
    cmf_32e38o.npz   active-space problem for TPSChem (CMF.export_fermicg: h0, h1, h2 with exact two-electron
                     integrals of the active orbitals, cluster_index, fspace, rdm1a/b, e_cmf, mo_coeff)
    orbitals.npz     doubly occupied, active and virtual orbitals with clusters and fspace, for tpsci_pdft.py
    rohf.chk         ROHF (reused when present), cmf_orbitals.npy (saved every ooCMF iteration: restarts),
    cmf.molden       active orbitals, pycmf.log (ooCMF iterations), summary.json
"""
import argparse, json, os, time
import numpy as np
from pyscf import gto, scf, lib, ao2mo
from pyscf.tools import molden
import pycmf

here = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--basis', default='def2-svp')
p.add_argument('--auxbasis', default='def2-universal-jkfit', help='density fitting of ROHF and ooCMF')
p.add_argument('--geometry', default=os.path.join(here, '..', 'data', 'cr2_pantazis.xyz'))
p.add_argument('--out', default=None, help='output directory (default: the basis name)')
p.add_argument('--rohf-chk', default=None, help='ROHF checkpoint to reuse (default: <out>/rohf.chk)')
p.add_argument('--guess-chk', default=None,
               help='ROHF checkpoint in another basis (e.g. svp/rohf.chk) whose density, projected, starts this ROHF')
p.add_argument('--conv-tol-grad', type=float, default=1e-5)
p.add_argument('--max-memory', type=int, default=12000, help='MB')
p.add_argument('--threads', type=int, default=None)
args = p.parse_args()
out = args.out or args.basis
os.makedirs(out, exist_ok=True)
if args.threads:
    lib.num_threads(args.threads)
t0 = time.time()


def log(*a):
    print('[%6.0f s]' % (time.time() - t0), *a, flush=True)


# ---------------------------------------------------------------- ROHF, high spin (S = 3)
mol = gto.M(atom=args.geometry, basis=args.basis, charge=3, spin=6, max_memory=args.max_memory, verbose=4,
            output=os.path.join(out, 'pyscf.log'))
log('%s: %d AOs, %d electrons' % (args.basis, mol.nao, mol.nelectron))
mf = scf.ROHF(mol).density_fit(auxbasis=args.auxbasis)
chk = args.rohf_chk or os.path.join(out, 'rohf.chk')
if os.path.exists(chk):
    mf.__dict__.update(lib.chkfile.load(chk, 'scf'))
    log('ROHF %.10f (%s)' % (mf.e_tot, chk))
else:
    mf.chkfile = chk
    mf.conv_tol, mf.max_cycle = 1e-9, 200
    dm0 = mf.from_chk(args.guess_chk) if args.guess_chk else None   # projected into this basis
    mf.kernel(dm0)
    log('ROHF %.10f  converged %s' % (mf.e_tot, mf.converged))
    if not mf.converged:
        raise RuntimeError('ROHF did not converge')

# ---------------------------------------------------------------- AO-cMF (32e, 38o)
labels = [['0 Cr 3d', '0 Cr 4d'], ['1 Cr 3d', '1 Cr 4d']] + [['%d O 2p' % i, '%d O 3p' % i] for i in (2, 3, 4)]
fspace = [(5, 2), (2, 5), (3, 3), (3, 3), (3, 3)]
cmf = pycmf.CMF(mf)
cmf.set_clusters_from_ao(labels, fspace, guess='spade')
cmf.algorithm = 'newton'
cmf.conv_tol_grad = args.conv_tol_grad
cmf.chkfile = os.path.join(out, 'cmf_orbitals.npy')
cmf.verbose, cmf.stdout = 4, open(os.path.join(out, 'pycmf.log'), 'a')
if os.path.exists(cmf.chkfile):
    cmf.set_mo_coeffs(np.load(cmf.chkfile))
    log('orbitals from %s' % cmf.chkfile)
log('active space: (%de, %do), clusters %s, %d doubly occupied, %d virtual' % (
    sum(map(sum, fspace)), cmf.Cact.shape[1], [len(c) for c in cmf.clusters], cmf.Cdoc.shape[1], cmf.Cvir.shape[1]))
e = cmf.kernel()
log('ooCMF E = %.10f  converged %s' % (e, cmf.converged))

# ---------------------------------------------------------------- integrals for TPSChem
# export_fermicg takes h2 from the density-fitted integrals of mf; replace it by the exact (aa|aa), as the
# Faraday integrals (ints_h2.npy), and h0/h1 by the exact core Coulomb/exchange
fn = os.path.join(out, 'cmf_32e38o.npz')
cmf.export_fermicg(fn)
d = dict(np.load(fn))
ncore, nact = int(d['ncore']), int(d['nact'])
C = d['mo_coeff']
Ccore, Cact = C[:, :ncore], C[:, ncore:ncore + nact]
mf_exact = scf.ROHF(mol)                                   # exact J/K for the core
dcore = 2 * Ccore @ Ccore.T
vj, vk = mf_exact.get_jk(mol, dcore)
hcore = mf.get_hcore()
d['h0'] = np.float64(mol.energy_nuc() + np.einsum('ij,ji->', dcore, hcore + .5 * vj - .25 * vk))
d['h1'] = Cact.T @ (hcore + vj - .5 * vk) @ Cact
d['h2'] = ao2mo.restore(1, ao2mo.kernel(mol, Cact), nact)
np.savez(fn, **d)
log('wrote %s (exact two-electron integrals)' % fn)

np.savez(os.path.join(out, 'orbitals.npz'), Cdoc=Ccore, Cact=Cact, Cvir=C[:, ncore + nact:],
         clusters=np.concatenate([np.full(len(c), i) for i, c in enumerate(cmf.clusters)]),
         fspace=np.array(fspace), basis=args.basis, auxbasis=args.auxbasis, geometry=os.path.abspath(args.geometry),
         e_cmf=e)
molden.from_mo(mol, os.path.join(out, 'cmf.molden'), Cact)
json.dump({'basis': args.basis, 'nao': mol.nao, 'e_rohf': mf.e_tot, 'e_cmf': e, 'converged': bool(cmf.converged),
           'fspace': fspace, 'clusters': [len(c) for c in cmf.clusters], 'ncore': ncore},
          open(os.path.join(out, 'summary.json'), 'w'), indent=1)
log('done')
