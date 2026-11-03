"""Build ao_cmf_200_active.ipynb (cells below) and execute it with the small test settings"""
import os, sys
import nbformat as nbf
from nbclient import NotebookClient

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip('\n')))

md(r'''
# AO-cMF with a 200-orbital active space

All-trans polyene C$_{100}$H$_{102}$ in 6-31G (202 atoms, 1104 AOs). Each carbon brings its 2p$_z$ and 3p$_z$ AOs to the
active space, **200 active orbitals**, in 25 clusters of four carbons (8 orbitals, 4 electrons each). The clusters are
meta-Löwdin orthogonalized AOs (`set_clusters_from_ao`, no orbital localization); the orbital optimization then
relaxes them together with the doubly occupied and virtual orbitals.

Steps: density-fitted RHF, orbital-optimized cMF (AO-cMF), QDPT2 between neighbouring clusters (pairs within a cutoff),
and cMF-NEVPT2 of the environment (frozen 1s). Every step writes checkpoints to `WORKDIR`, so the notebook can be
rerun after a stop and continues where it was (the cMF orbitals are saved after every iteration).

**Resources for the 200-orbital run** (the cell below prints them for the chosen settings): the 3-index integrals
take 29 GB, kept in memory when 0.8 `MAX_MEMORY` is larger, otherwise streamed from disk. The active space only adds
its own 3-index factors (L|pq), 2 GB: the full (aa|aa) (13 GB) is never formed. cMF and QDPT2 run on a 64 GB node
(`MAX_MEMORY = 50000`, in MB); the Newton optimizer keeps (L|i nu) of the doubly occupied and active orbitals (27 GB)
when it fits in half of the free `MAX_MEMORY` and recomputes it in every Hessian product otherwise (about 1.5x
slower). NEVPT2 adds (L|p t) for all orbitals p and active t (11 GB) and (i a|L) (6 GB), so
use 128 GB (`MAX_MEMORY = 100000`) for that step. The 200-orbital cMF should take of the order of 10 h on 4 cores
and scale down with the number of cores (estimated from the timings of the STO-3G polyene in the README); the QDPT2
pairs take about a minute each and NEVPT2 is dominated by its core-core/virtual-virtual term. For a batch job:
`jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 ao_cmf_200_active.ipynb` (or
`jupyter nbconvert --to script` and run the `.py`).

The saved outputs are the setup of the (100e, 200o) calculation; the calculation itself needs the node above.
`TEST_RUN = True` runs the same workflow for C$_{16}$H$_{18}$, (16e, 32o) in 4 clusters, in a few minutes on 2 cores:

| step | E (Eh) |
| --- | --- |
| RHF | −616.0069413 |
| AO-cMF: Newton 6 steps (12 Hessian products) in 38 s; L-BFGS 22 iterations in 78 s | −616.2366445 |
| + QDPT2 between the 3 cluster pairs | −616.2547050 |
| + cMF-NEVPT2 (order 1, frozen 1s) | −617.4157630 |
''')

code(r'''
import os
os.environ.setdefault('OMP_NUM_THREADS', str(os.cpu_count()))     # before NumPy/PySCF are imported; in a batch job
                                                                 # set it to the job's cores (SLURM: $SLURM_CPUS_PER_TASK)

TEST_RUN = False             # True: 16 carbons, (16e, 32o), a 5-minute check of the setup
NC = 16 if TEST_RUN else 100 # carbons; active orbitals = NC * len(SHELLS)
BASIS = '6-31g'
SHELLS = ['2pz', '3pz']      # AOs of each carbon in the active space
CARBONS_PER_CLUSTER = 4
AUXBASIS = 'weigend'
MAX_MEMORY = 8000 if TEST_RUN else 100000      # MB for PySCF: 50000 is enough for cMF and QDPT2
CONV_TOL_GRAD = 1e-4         # orbital gradient of the cMF orbital optimization
ALGORITHM = 'newton'         # cMF orbital optimizer: 'newton' (trust-region Newton) or 'lbfgs'
PAIR_CUTOFF = 6.0            # Angstrom, cluster pairs for QDPT2
NEVPT2_ORDER = 1             # 1: single clusters (cheap); 2 adds the cluster pairs (16-orbital NEVPT2 each, expensive)
WORKDIR = 'polyene_C%d_%s' % (NC, BASIS)
os.makedirs(WORKDIR, exist_ok=True)
print('threads:', os.environ['OMP_NUM_THREADS'], '  work directory:', WORKDIR)
''')

md(r'''
## Molecule, clusters and resource estimate
''')
code(r'''
import json, time
import numpy as np
from pyscf import gto, scf, lib, df
import pycmf
print('pycmf from', os.path.dirname(pycmf.__file__))
# an older pycmf ignores cmf.algorithm and always runs L-BFGS
assert hasattr(pycmf.CMF, 'gen_g_hop'), ('this pycmf has no Newton optimizer: git pull, pip install -e . in the '
                                         'repository, and restart the kernel')

def polyene(nc):
    """all-trans polyene C_nc H_(nc+2) in the xy plane, carbons first (C=C 1.35, C-C 1.45, C-H 1.09 A)"""
    c = [np.zeros(3)]
    for k in range(nc - 1):
        L = 1.35 if k % 2 == 0 else 1.45
        ang = np.radians(30) * (1 if k % 2 == 0 else -1)
        c.append(c[-1] + L*np.array([np.cos(ang), np.sin(ang), 0]))
    c = np.array(c)
    rot = lambda v, a: np.array([v[0]*np.cos(a) - v[1]*np.sin(a), v[0]*np.sin(a) + v[1]*np.cos(a), 0])
    h = []
    for i in range(nc):
        if i in (0, nc - 1):
            u = c[1] - c[0] if i == 0 else c[-2] - c[-1]
            u = u / np.linalg.norm(u)
            h += [c[i] + 1.09*rot(u, np.radians(120)), c[i] + 1.09*rot(u, np.radians(-120))]
        else:
            b = (c[i-1] - c[i])/np.linalg.norm(c[i-1] - c[i]) + (c[i+1] - c[i])/np.linalg.norm(c[i+1] - c[i])
            h.append(c[i] - 1.09*b/np.linalg.norm(b))
    return [['C', tuple(x)] for x in c] + [['H', tuple(x)] for x in h]

mol = gto.M(atom=polyene(NC), basis=BASIS, max_memory=MAX_MEMORY, verbose=0)
labels = [['%d C %s' % (a, s) for a in range(k, k + CARBONS_PER_CLUSTER) for s in SHELLS]
          for k in range(0, NC, CARBONS_PER_CLUSTER)]
ncl = len(labels)
fspace = [(CARBONS_PER_CLUSTER//2, CARBONS_PER_CLUSTER//2)] * ncl     # one pi electron per carbon
nact = NC * len(SHELLS)

naux = df.addons.make_auxmol(mol, AUXBASIS).nao
gb = lambda n: 8*n/1e9
nocc = mol.nelectron//2 - sum(map(sum, fspace))//2
print('%d atoms, %d AOs, %d auxiliary functions' % (mol.natm, mol.nao, naux))
print('active space: (%de, %do) in %d clusters of %d orbitals, %d doubly occupied, %d virtual'
      % (sum(map(sum, fspace)), nact, ncl, nact // ncl, nocc, mol.nao - nocc - nact))
print('memory (GB): 3-index integrals %.1f, active (L|pq) %.1f, Newton cache (L|i nu) %.1f, NEVPT2 (L|pt) + (ia|L) %.1f,'
      ' given to PySCF %.0f' % (gb(naux*mol.nao*(mol.nao + 1)/2), gb(naux*nact**2), gb(naux*(nocc + nact)*mol.nao),
                                gb(naux*mol.nao*nact) + gb((nocc - NC)*(mol.nao - nocc - nact)*naux), MAX_MEMORY/1000))
''')

md(r'''
## Density-fitted RHF
''')
code(r'''
t = time.time()
mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
mf.chkfile = os.path.join(WORKDIR, 'rhf.chk')
if os.path.exists(mf.chkfile):
    mf.__dict__.update(lib.chkfile.load(mf.chkfile, 'scf'))
    print('RHF %.10f (from %s)' % (mf.e_tot, mf.chkfile))
else:
    mf.conv_tol = 1e-9
    mf.kernel()
    print('RHF %.10f  converged %s  %.0f s' % (mf.e_tot, mf.converged, time.time() - t))
''')

md(r'''
## AO-cMF

The cluster orbitals start as the orthogonalized 2p$_z$/3p$_z$ AOs and the doubly occupied orbitals as the mean-field
density in the rest, so the first energy is above the RHF one: the optimization moves the $\pi$ character back into
the occupied space and correlates each cluster exactly.

`ALGORITHM` selects the optimizer. `'lbfgs'`: L-BFGS, one energy and gradient per iteration (plus line-search
evaluations). `'newton'`: trust-region Newton with the orbital Hessian at fixed cluster RDMs, each step solved by
conjugate gradients with Hessian-vector products (each about the cost of 1.5-2 gradients), so far fewer but more
expensive iterations; it copes with the very soft orbital modes that make L-BFGS crawl. The two optimizers write
separate files, `cmf.log` / `cmf_orbitals.npy` and `cmf_newton.log` / `cmf_orbitals_newton.npy` in `WORKDIR` (the
orbitals after every iteration; rerunning this cell continues from them), so both can be run and compared below.
The iterations are printed below the cell and written to the log.
''')
code(r'''
import sys

class Tee:
    """Write to several streams: the iterations go to the notebook and to the log file"""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for f in self.streams:
            f.write(s)
    def flush(self):
        for f in self.streams:
            f.flush()

tag = '' if ALGORITHM == 'lbfgs' else '_' + ALGORITHM      # files of this optimizer
cmf = pycmf.CMF(mf)
cmf.set_clusters_from_ao(labels, fspace, guess='ao')
cmf.algorithm = ALGORITHM
cmf.conv_tol_grad = CONV_TOL_GRAD
cmf.conv_tol = 1e-8
cmf.chkfile = os.path.join(WORKDIR, 'cmf_orbitals%s.npy' % tag)
if os.path.exists(cmf.chkfile):
    cmf.set_mo_coeffs(np.load(cmf.chkfile))
    print('continuing from', cmf.chkfile)
cmf.verbose = 4                                       # iterations; 5 adds every CMF-CI sweep
logfile = open(os.path.join(WORKDIR, 'cmf%s.log' % tag), 'a')
cmf.stdout = Tee(sys.stdout, logfile)
t = time.time()
e_cmf = cmf.kernel()
logfile.close()
cmf.stdout, cmf.verbose = mf.stdout, 0
with open(os.path.join(WORKDIR, 'cmf%s_time.txt' % tag), 'a') as f:     # wall time of each run of this cell
    f.write('%.1f\n' % (time.time() - t))
g = np.linalg.norm(cmf.orbital_gradient()[cmf.rotation_mask()])     # as in the convergence test
print('%s cMF %.10f  (RHF %.10f, %.1f mEh lower)  converged %s  |g| = %.1e  %.0f s'
      % (ALGORITHM, e_cmf, mf.e_tot, 1000*(mf.e_tot - e_cmf), cmf.converged, g, time.time() - t))
''')

md(r'''
### Newton against L-BFGS

From the logs of the two optimizers in `WORKDIR` (run the cMF cell once with each `ALGORITHM`): iterations, Hessian
products of Newton, final energy and gradient, and the wall time of the cMF cell (summed over restarts; a log
written before this notebook recorded times has none). The AO start can lead the two optimizers to different local
minima (clusters polarized differently), so compare the final energies as well.
''')
code(r'''
import re
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None                                 # the table only

def read_log(name):
    """(E, |g|) of every iteration of a cmf log (restarts joined), the number of Hessian products, and the wall time"""
    path = os.path.join(WORKDIR, name + '.log')
    if not os.path.exists(path):
        return None
    hist, ncg = [], 0
    for line in open(path):
        if line.startswith('ooCMF iter'):
            if int(line.split()[2]) == 0 and hist:
                continue                       # a restart repeats the last iteration
            hist.append((float(re.search(r'E = (\S+)', line).group(1)), float(re.search(r'\|g\| = (\S+)', line).group(1))))
            m = re.search(r'CG (\d+)', line)
            ncg += int(m.group(1)) if m else 0
    tfile = os.path.join(WORKDIR, name + '_time.txt')
    wall = sum(map(float, open(tfile).read().split())) if os.path.exists(tfile) else None
    return np.array(hist), ncg, wall

runs = {alg: read_log(name) for alg, name in (('L-BFGS', 'cmf'), ('Newton', 'cmf_newton'))}
runs = {k: v for k, v in runs.items() if v is not None and len(v[0])}
print('%-8s %10s %16s %18s %10s %10s' % ('', 'iterations', 'Hessian products', 'final E', '|g|', 'wall time'))
for alg, (hist, ncg, wall) in runs.items():
    print('%-8s %10d %16s %18.10f %10.1e %10s' % (alg, len(hist) - 1, ncg if alg == 'Newton' else '-', hist[-1, 0],
          hist[-1, 1], '%.2f h' % (wall/3600) if wall else 'n/a'))
if runs and plt is not None:
    e_min = min(hist[:, 0].min() for hist, ncg, wall in runs.values())
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for alg, (hist, ncg, wall) in runs.items():
        ax[0].semilogy(hist[:, 0] - e_min + 1e-10, label=alg)
        ax[1].semilogy(hist[:, 1], label=alg)
    ax[0].set(xlabel='iteration', ylabel='E - E$_{min}$ (Eh)')
    ax[1].set(xlabel='iteration', ylabel='|g|')
    ax[1].axhline(CONV_TOL_GRAD, color='gray', ls=':')
    ax[0].legend()
    plt.tight_layout()
''')

md(r'''
### The clusters

Natural occupations show how strongly correlated each cluster is (2 and 0 for a closed-shell determinant); the end
clusters are usually the most correlated in a polyene.
''')
code(r'''
for i, (ci, d) in enumerate(zip(cmf.clusters, cmf.cluster_1rdm)):
    n = np.linalg.eigvalsh(d)[::-1]
    print('cluster %2d  N = %.4f  E = %.8f  natural occupations %s' % (i, np.trace(d), cmf.cluster_energies[i],
          ' '.join('%.3f' % x for x in n[:4]) + ' ... ' + ' '.join('%.3f' % x for x in n[-2:])))
print('<S^2> of the product state: %.6f' % cmf.spin_square()[0])
''')

md(r'''
## QDPT2 between neighbouring clusters

`pt2_mbe(order=2, cutoff)` adds the second-order correlation (charge transfer and dispersion) of each pair of clusters
whose centroids are within `PAIR_CUTOFF`, in the mean field of the others. Each pair is 16 orbitals here.
''')
code(r'''
t = time.time()
e_pt2, inc = cmf.pt2_mbe(order=2, cutoff=PAIR_CUTOFF)
print('%d pairs within %.1f A: E(cMF + QDPT2) = %.10f  correlation between clusters %.6f  %.0f s'
      % (len(inc), PAIR_CUTOFF, e_pt2, e_pt2 - e_cmf, time.time() - t))
print('largest pair increments:', {k: round(float(v), 6) for k, v in sorted(inc.items(), key=lambda kv: kv[1])[:5]})
''')

md(r'''
## cMF-NEVPT2 of the environment

Dynamic correlation of the doubly occupied and virtual orbitals (the carbon 1s are frozen), expanded over the
clusters. The core-core/virtual-virtual term dominates the cost, $O(n_{\rm core}^2 n_{\rm virt}^2 N_{\rm aux})$: about an
hour on 4 cores for the 200-orbital run, much less on a large node.
''')
code(r'''
from pyscf.data.elements import chemcore
from pycmf import mrpt
ncore = chemcore(mol)
fock = mrpt._prepare(cmf)['fock']
e, u = np.linalg.eigh(cmf.Cdoc.T @ fock @ cmf.Cdoc)          # freeze the lowest doubly occupied orbitals (C 1s)
C = cmf.Cdoc @ u
cmf.set_Cfrz(C[:, :ncore])
cmf.set_Cdoc(C[:, ncore:])
t = time.time()
e_nev, inc_nev = cmf.nevpt2_mbe(order=NEVPT2_ORDER, cutoff=PAIR_CUTOFF)
print('%d frozen, %d correlated doubly occupied, %d virtual orbitals' % (ncore, cmf.Cdoc.shape[1], cmf.Cvir.shape[1]))
print('cMF-NEVPT2 (order %d): %.8f  (core-core/virtual-virtual %.8f)  %.0f s' % (NEVPT2_ORDER, e_nev, inc_nev['sijrs'], time.time() - t))
''')

md(r'''
## Summary
''')
code(r'''
results = dict(nc=NC, basis=BASIS, nao=mol.nao, nact=nact, nclusters=ncl, e_rhf=mf.e_tot, e_cmf=e_cmf,
               cmf_converged=bool(cmf.converged), e_qdpt2=e_pt2, e_nevpt2=e_nev, e_total=e_pt2 + e_nev)
json.dump(results, open(os.path.join(WORKDIR, 'results.json'), 'w'), indent=1)
for k in ('e_rhf', 'e_cmf', 'e_qdpt2', 'e_total'):
    print('%-8s %.10f' % (k, results[k]))
print('E(cMF) + QDPT2 between clusters + NEVPT2 of the environment = %.10f' % results['e_total'])
''')

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                  'language_info': {'name': 'python'}}
out, path = sys.argv[1], sys.argv[2]
if os.environ.get('NB_SETTINGS'):          # test runs: replace settings lines, e.g. "TEST_RUN = True;ALGORITHM = 'lbfgs'"
    for line in os.environ['NB_SETTINGS'].split(';'):
        key = line.split('=')[0].strip()
        src = nb['cells'][1]['source'].splitlines()
        nb['cells'][1]['source'] = '\n'.join(line + '   # (test run)' if l.startswith(key + ' ') else l for l in src)
nexec = int(sys.argv[3]) if len(sys.argv) > 3 else len(cells)      # execute the first nexec cells only
run = nbf.v4.new_notebook()
run['cells'], run['metadata'] = nb['cells'][:nexec], nb['metadata']
NotebookClient(run, timeout=7200, kernel_name='python3', resources={'metadata': {'path': path}}).execute()
nb['cells'][:nexec] = run['cells']
nbf.write(nb, out)
print('written', out)
