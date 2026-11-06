"""
[Cr2(OH)3(tmtacn)2]3+: add the tmtacn donor atoms to the large active space of cr2_active_space_selection.py.

Each of the six donor N becomes a cluster with its lone pair (the sigma donor to Cr), its three N-C sigma bonds and four
correlating orbitals (mostly the N-C sigma*): (4, 4) in 8 orbitals. A cluster per tmtacn would be (12, 12) in 24
orbitals, too large for cluster FCI; per N the clusters do not overlap, since every carbon is bonded to one N only.

Starting point: the converged (30e, 52o) ooCMF (Cr 3d4d4s4p, O 2s2p3s3p; cmf_big.npy). Its doubly occupied orbitals are
localized (Pipek-Mezey); a localized valence orbital (not 1s) goes to a donor N when that N carries its largest
meta-Lowdin population; each N must get four (lone pair + three N-C bonds, possibly mixed by the localization). The four correlating
orbitals of each N are the virtual orbitals that overlap most with the valence (2s2p, --vir-shells) meta-Lowdin AOs of the
N and its three carbons (SVD of the projection, one cluster after the other): with all AOs many virtual orbitals of
polarization functions would overlap completely, and the choice would be arbitrary. ooCMF then optimizes all orbitals in (78e, 100o).

    python cr2_ligand_clusters.py --rohf-chk rohf.chk --big cmf_big.npy

The orbitals are saved after every ooCMF iteration (cmf_lig.npy, resumed when present), the iterations go to
pycmf_lig.log, and the natural occupations and entropies of all clusters to ligand_clusters.json.
"""
import argparse, json, os, time
import numpy as np
from pyscf import gto, scf, lib, lo
import pycmf

here = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--rohf-chk', default='rohf.chk')
p.add_argument('--big', default='cmf_big.npy', help='orbitals of the converged (30e, 52o) ooCMF')
p.add_argument('--geometry', default=os.path.join(here, 'data', 'cr2_pantazis.xyz'))
p.add_argument('--basis', default='def2-svp')
p.add_argument('--auxbasis', default='def2-universal-jkfit')
p.add_argument('--cr-shells', nargs='+', default=['3d', '4d', '4s', '4p'])
p.add_argument('--o-shells', nargs='+', default=['2s', '2p', '3s', '3p'])
p.add_argument('--nvir', type=int, default=4, help='correlating orbitals per N cluster')
p.add_argument('--vir-shells', nargs='+', default=['2s', '2p'],
               help='AOs of the N and its carbons that the correlating orbitals are projected on (valence: sigma*)')
p.add_argument('--c-pop', type=float, default=0.2, help='carbon population that makes a localized orbital a N-C bond')
p.add_argument('--conv-tol-grad', type=float, default=1e-5)
p.add_argument('--max-memory', type=int, default=12000)
args = p.parse_args()
t0 = time.time()


def log(*a):
    print('[%6.0f s]' % (time.time() - t0), *a, flush=True)


mol = gto.M(atom=args.geometry, basis=args.basis, charge=3, spin=6, max_memory=args.max_memory, verbose=0)
mf = scf.ROHF(mol).density_fit(auxbasis=args.auxbasis)
mf.__dict__.update(lib.chkfile.load(args.rohf_chk, 'scf'))
log('ROHF %.10f' % mf.e_tot)

# ---------------------------------------------------------------- the (30e, 52o) state
labels = [['0 Cr %s' % s for s in args.cr_shells], ['1 Cr %s' % s for s in args.cr_shells]]
labels += [['%d O %s' % (ia, s) for s in args.o_shells] for ia in (2, 3, 4)]
nel_o = (2 if '2s' in args.o_shells else 0) + (6 if '2p' in args.o_shells else 0)
fspace = [(3, 0), (0, 3)] + [(nel_o // 2, nel_o // 2)] * 3
big = pycmf.CMF(mf)
big.set_clusters_from_ao(labels, fspace, guess='ao')
big.set_mo_coeffs(np.load(args.big))
Cdoc, Cact, Cvir = big.Cdoc, big.Cact, big.Cvir
S = big.S

# ---------------------------------------------------------------- donor N and their carbons
sym = [mol.atom_symbol(i) for i in range(mol.natm)]
R = mol.atom_coords()
dist = np.linalg.norm(R[:, None] - R[None], axis=2) * lib.param.BOHR
donors = [i for i in range(mol.natm) if sym[i] == 'N']
carbons = {n: [j for j in range(mol.natm) if sym[j] == 'C' and dist[n, j] < 1.7] for n in donors}
log('donor N and bonded C: %s' % carbons)

# ---------------------------------------------------------------- localized doubly occupied orbitals
X = lo.orth_ao(mol, 'meta_lowdin', s=S)
ao_atom = np.array([int(l.split()[0]) for l in mol.ao_labels()])
loc = lo.PM(mol, Cdoc)
loc.pop_method = 'meta_lowdin'
Cloc = loc.kernel()
c = X.T @ S @ Cloc
pop = np.array([np.sum(c[ao_atom == a]**2, axis=0) for a in range(mol.natm)])      # (natm, norb)
core = np.array([l.split()[2].startswith('1s') for l in mol.ao_labels()])             # 1s AOs (meta-Lowdin)
core_pop = np.sum(c[core]**2, axis=0)
occ_of = {n: [] for n in donors}
for k in range(Cloc.shape[1]):
    a = int(np.argmax(pop[:, k]))
    if a in occ_of and core_pop[k] < 0.5:                       # valence orbitals of a donor N (not its 1s)
        cpop = {j: pop[j, k] for j in carbons[a]}
        jc = max(cpop, key=cpop.get)
        # labels for the log only: Pipek-Mezey may mix the lone pair with a N-C bond ("rabbit ears"); the four
        # orbitals span the lone pair and the three bonds either way
        occ_of[a].append((k, 'LP' if cpop[jc] < args.c_pop else 'NC%d' % jc, pop[a, k], cpop[jc]))
for n in donors:
    log('N %d: %s' % (n, ', '.join('%s (N %.2f, C %.2f)' % (t, pn, pc) for _, t, pn, pc in occ_of[n])))
    if len(occ_of[n]) != 4:
        raise RuntimeError('N %d: expected 4 valence localized orbitals (lone pair + 3 N-C bonds), got %d'
                           % (n, len(occ_of[n])))

# ---------------------------------------------------------------- correlating virtual orbitals per N
Crest = Cvir
valence = np.array([any(l.split()[2].startswith(sh) for sh in args.vir_shells) for l in mol.ao_labels()])
blocks, ligand_fspace = [], []
for n in donors:
    atoms = [n] + carbons[n]
    XK = X[:, np.isin(ao_atom, atoms) & valence]
    u, sv, vt = np.linalg.svd(XK.T @ S @ Crest, full_matrices=True)
    Cv = Crest @ vt[:args.nvir].T
    Crest = Crest @ vt[args.nvir:].T
    Co = Cloc[:, [k for k, _, _, _ in occ_of[n]]]
    blocks.append(np.hstack((Co, Cv)))
    ligand_fspace.append((Co.shape[1], Co.shape[1]))
    log('N %d: correlating orbitals with singular values %s' % (n, np.round(sv[:args.nvir], 3).tolist()))
used = [k for n in donors for k, _, _, _ in occ_of[n]]
Cdoc_new = Cloc[:, [k for k in range(Cloc.shape[1]) if k not in used]]

# ---------------------------------------------------------------- (78e, 100o) ooCMF
clusters = [list(c) for c in big.clusters]
n0 = Cact.shape[1]
for b in blocks:
    clusters.append(list(range(n0, n0 + b.shape[1])))
    n0 += b.shape[1]
cmf = pycmf.CMF(mf)
cmf.set_Cdoc(Cdoc_new)
cmf.set_Cact(np.hstack([Cact] + blocks), clusters, list(big.fspace) + ligand_fspace)
cmf.set_Cvir(Crest)
C = cmf.get_mo_coeffs()
assert np.allclose(C.T @ S @ C, np.eye(C.shape[1]), atol=1e-8)
assert 2*cmf.Cdoc.shape[1] + sum(map(sum, cmf.fspace)) == mol.nelectron
cmf.algorithm = 'newton'
cmf.conv_tol_grad = args.conv_tol_grad
cmf.chkfile = 'cmf_lig.npy'
cmf.verbose, cmf.stdout = 4, open('pycmf_lig.log', 'a')
if os.path.exists('cmf_lig.npy'):
    cmf.set_mo_coeffs(np.load('cmf_lig.npy'))
    log('orbitals from cmf_lig.npy')
log('active space: (%de, %do), clusters %s, %d doubly occupied, %d virtual' % (
    sum(map(sum, cmf.fspace)), cmf.Cact.shape[1], [len(c) for c in cmf.clusters], cmf.Cdoc.shape[1],
    cmf.Cvir.shape[1]))
e = cmf.kernel()
log('ooCMF E = %.10f  converged %s  (big (30e, 52o): %.10f)' % (e, cmf.converged, big.e_tot if big.e_tot else np.nan))

ent = cmf.orbital_entropies()
out = {'e': e, 'converged': bool(cmf.converged), 'fspace': cmf.fspace, 'clusters': [len(c) for c in cmf.clusters],
       'donors': donors, 'occ': [x['occ'].tolist() for x in ent], 'entropy': [x['entropy'].tolist() for x in ent]}
for i, x in enumerate(ent):
    log('cluster %2d %s: occupations %s' % (i, tuple(cmf.fspace[i]), np.round(x['occ'], 4).tolist()))
json.dump(out, open('ligand_clusters.json', 'w'), indent=1, default=float)
