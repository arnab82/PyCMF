"""
Singlet fission in acene dimers: TPSCI + L-PDFT with one cluster per monomer (its frontier pi orbitals), diabatic
LE / CT / TT analysis, direct vs CT-mediated couplings.

Geometry: idealized planar acene (C-C 1.40 A, C-H 1.08 A) in the xy plane, long axis x; the second monomer stacked at
--distance along z, slipped by --slip-long (x) and --slip-short (y).

    python acene_dimer.py prepare --rings 4 --out tc --slip-long 1.2
    julia --project=/path/to/TPSChem.jl -t auto tpsci_states.jl tc/cmf.npz tc/states.npz
    python acene_dimer.py analyze --out tc                                  # same geometry arguments

Clusters: the dimer RHF occupied and virtual spaces are each split between the monomers by SVD projection on the
monomer AOs, Fock-canonicalized within each monomer, and the --nfront highest occupied and lowest virtual orbitals of
each monomer form its cluster ((n, n) electrons); the rest are doubly occupied / virtual. With --oo, ooCMF optimizes
all orbitals afterwards.
"""
import argparse, os, sys
import numpy as np
from pyscf import gto, scf, lib
import pycmf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sf_analysis

p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('step', choices=['prepare', 'analyze'])
p.add_argument('--out', default='acene')
p.add_argument('--rings', type=int, default=4, help='2 naphthalene, 3 anthracene, 4 tetracene, 5 pentacene')
p.add_argument('--distance', type=float, default=3.5)
p.add_argument('--slip-long', type=float, default=0.0)
p.add_argument('--slip-short', type=float, default=0.0)
p.add_argument('--basis', default='6-31g')
p.add_argument('--nfront', type=int, default=2, help='frontier occupied (and virtual) orbitals per monomer')
p.add_argument('--oo', action='store_true', help='optimize all orbitals with ooCMF')
p.add_argument('--functionals', default='tPBE,ftPBE,tBLYP,tPBE0')
p.add_argument('--no-check', action='store_true', help='skip the CASCI L-PDFT check (large active spaces)')
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)


def acene(n, a=1.40, ch=1.08):
    """Carbon and hydrogen positions of a linear acene with n rings, in the xy plane, long axis x"""
    s3 = np.sqrt(3)
    centers = [((i - (n - 1) / 2) * s3 * a, 0.0) for i in range(n)]
    carbons = []
    for cx, cy in centers:
        for ang in np.radians([90, 150, 210, 270, 330, 30]):
            v = np.array([cx + a * np.cos(ang), cy + a * np.sin(ang)])
            if not any(np.linalg.norm(v - c) < 1e-3 for c in carbons):
                carbons.append(v)
    carbons = np.array(carbons)
    hydrogens = []
    for c in carbons:
        nb = [d for d in carbons if 0 < np.linalg.norm(d - c) < 1.1 * a]
        if len(nb) == 2:
            u = 2 * c - nb[0] - nb[1]
            hydrogens.append(c + ch * u / np.linalg.norm(u))
    return carbons, np.array(hydrogens)


C, H = acene(args.rings)
shift = np.array([args.slip_long, args.slip_short, args.distance])
atoms = []
for off in (np.zeros(3), shift):
    atoms += [('C', np.r_[c, 0] + off) for c in C] + [('H', np.r_[h, 0] + off) for h in H]
nat = len(C) + len(H)
mol = gto.M(atom=[(s, tuple(x)) for s, x in atoms], basis=args.basis, verbose=0, max_memory=8000)
mf = scf.RHF(mol).density_fit()
chk = os.path.join(args.out, 'rhf.chk')
if os.path.exists(chk):
    mf.__dict__.update(lib.chkfile.load(chk, 'scf'))
else:
    mf.chkfile = chk
    mf.run(conv_tol=1e-10)
print('%d-ring acene dimer, %d AOs, RHF %.10f' % (args.rings, mol.nao, mf.e_tot))

# ---------------------------------------------------------------- frontier orbitals of each monomer
S, F = mf.get_ovlp(), mf.get_fock()
Cocc, Cvir = mf.mo_coeff[:, mf.mo_occ > 0], mf.mo_coeff[:, mf.mo_occ == 0]
from pyscf import lo
X = lo.orth_ao(mol, 'meta_lowdin', s=S)
ao_atom = np.array([int(l.split()[0]) for l in mol.ao_labels()])
onA = ao_atom < nat


def split(Cs, nA):
    """Split an orbital space between the monomers: the nA orbitals with the largest projection on monomer A"""
    u, sv, vt = np.linalg.svd(X[:, onA].T @ S @ Cs, full_matrices=True)
    return Cs @ vt[:nA].T, Cs @ vt[nA:].T


def canon(Cs):
    e, u = np.linalg.eigh(Cs.T @ F @ Cs)
    return Cs @ u, e


nocc_mono = Cocc.shape[1] // 2
OA, OB = split(Cocc, nocc_mono)
VA, VB = split(Cvir, Cvir.shape[1] // 2)
k = args.nfront
blocks, doc, vir = [], [], []
for O, V in ((OA, VA), (OB, VB)):
    O, eo = canon(O)
    V, ev = canon(V)
    blocks.append(np.hstack((O[:, -k:], V[:, :k])))
    doc.append(O[:, :-k])
    vir.append(V[:, k:])
    print('monomer frontier orbital energies (eV): occ %s, vir %s' % (np.round(eo[-k:] * sf_analysis.EV, 3),
                                                                     np.round(ev[:k] * sf_analysis.EV, 3)))
cmf = pycmf.CMF(mf)
cmf.set_Cdoc(np.hstack(doc))
cmf.set_Cact(np.hstack(blocks), [list(range(2 * k)), list(range(2 * k, 4 * k))], [(k, k), (k, k)])
cmf.set_Cvir(np.hstack(vir))
orbfile = os.path.join(args.out, 'cmf_orbitals.npy')

if args.step == 'prepare':
    cmf.algorithm = 'newton'
    e = cmf.kernel() if args.oo else cmf.cmf_ci()
    np.save(orbfile, cmf.get_mo_coeffs())
    cmf.export_fermicg(os.path.join(args.out, 'cmf.npz'))
    print('cMF %.10f (%s), (%de, %do) -> %s/cmf.npz' % (e, 'ooCMF' if args.oo else 'frontier orbitals',
                                                     4 * k, 4 * k, args.out))
    raise SystemExit

cmf.set_mo_coeffs(np.load(orbfile))
cmf.cmf_ci()
sf_analysis.analyze(cmf, args.out, (2 * k, 2 * k), args.functionals.split(','), check=not args.no_check)
