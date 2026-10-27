"""
Exchange coupling of [Cr2(OH)3(tmtacn)2]3+ (Pantazis) with QDPT2 and cMF-PDFT, in the 19-orbital model of
github.com/arnab82/cMF_data (bimetallics/cr2_pantazis/3d_2p): def2-SVP, Cr 3d and bridging O 2p clusters from SPADE
(the repository's orbital_partitioning.py), environment frozen as in its cmf.jl.

    CMF_DATA=/path/to/cMF_data python cr2_pantazis_pdft.py

The ROHF (665 AOs) takes about half an hour on 4 cores; the QDPT2/PDFT part about an hour.
"""
import os, sys, json, time
import numpy as np
from pyscf import gto, scf, lib
import pycmf
from pycmf import pdft

cm = 219474.6313705
data = os.path.join(os.environ.get('CMF_DATA', 'cMF_data'), 'bimetallics', 'cr2_pantazis', '3d_2p')
sys.path.insert(0, data)
import orbital_partitioning as op

nb = json.load(open(os.path.join(data, 'scf.ipynb')))
cell = next(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code' and 'molecule = """' in ''.join(c['source']))
mol = gto.M(atom=cell.split('molecule = """')[1].split('"""')[0], basis='def2-svp', charge=3, spin=6, verbose=0,
            max_memory=10000)
mf = scf.ROHF(mol).density_fit(auxbasis='def2-universal-jkfit')
mf.chkfile = 'cr2_rohf.chk'
if os.path.exists(mf.chkfile):
    mf.__dict__.update(lib.chkfile.load(mf.chkfile, 'scf'))
else:
    mf.conv_tol = 1e-9
    mf.kernel()
print('ROHF %.10f' % mf.e_tot)

# active space as in the repository: SPADE onto Cr 3d and O 2p, then onto each fragment
S = mf.get_ovlp()
Cdocc, Csing, Cvirt = (mf.mo_coeff[:, mf.mo_occ == k] for k in (2, 1, 0))
frags = [[], [], [], [], []]
for i, ao in enumerate(mol.ao_labels(fmt=False)):
    if (ao[0] in (0, 1) and ao[2] == '3d') or (ao[0] in (2, 3, 4) and ao[2] == '2p'):
        frags[ao[0]].append(i)
I = np.eye(mol.nao)
(Oact, Sact, Vact), (Cenv, Cerr, _) = op.svd_subspace_partitioning((Cdocc, Csing, Cvirt), I[:, sum(frags, [])], S)
fspace, clusters, Cfrags, k = [], [], [], 0
for f in frags:
    (Of, Sf, Vf), _ = op.svd_subspace_partitioning((Oact, Sact, Vact), I[:, f], S)
    Cfrags.append(np.hstack((Of, Sf, Vf)))
    fspace.append((Of.shape[1] + Sf.shape[1], Of.shape[1]))
    clusters.append(list(range(k, k + Cfrags[-1].shape[1])))
    k += Cfrags[-1].shape[1]
fspace[1] = fspace[1][::-1]                  # antiparallel M_s on the second Cr

cmf = pycmf.CMF(mf)
cmf.set_Cfrz(Cenv)                           # environment frozen, as in the repository's active-space cMF
cmf.set_Cdoc(np.zeros((mol.nao, 0)))
cmf.set_Cact(np.hstack(op.sym_ortho(Cfrags, S)), clusters, fspace)
cmf.set_Cvir(np.zeros((mol.nao, 0)))
print('cMF %.10f  (cmf.jl, exact integrals: -3343.929813602433)' % cmf.kernel())

np.save('cr2_cmf_orbitals.npy', cmf.get_mo_coeffs())

ots = {x: pdft.make_ot(mol, x, grids_level=3) for x in ('tPBE', 'ftPBE')}
for x, ot in ots.items():
    print('spin-coupled cMF states, %s: J = %.1f cm-1' % (x, pdft.spin_ladder(cmf, ot=ot)[1]*cm), flush=True)

# QDPT2 first-order RDMs of the spin states: the Cr-Cr pair alone, and expanded over the pair and the three
# Cr-O-Cr triples; E_ot is evaluated once per spin state on the assembled RDMs
for label, subsets in (('Cr-Cr pair', [(0, 1)]), ('pair + 3 Cr-O-Cr triples', [(0, 1), (0, 1, 2), (0, 1, 3), (0, 1, 4)])):
    t = time.time()
    rdms = pdft.mbe_rdm12(cmf, subsets, verbose=4)
    js = {x: pdft.mbe_spin_ladder(cmf, ot=ot, rdms=rdms)[1]*cm for x, ot in ots.items()}
    print('%-26s' % label, '  '.join('J(cMF-PDFT %s) = %.1f' % kv for kv in js.items()), ' (%.0f s)' % (time.time() - t), flush=True)
