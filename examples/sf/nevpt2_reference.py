"""
Wavefunction reference for the L-PDFT energetics of an acene dimer: SC-NEVPT2 on the lowest CASCI singlets in the
cMF orbitals of acene_dimer.py (same geometry arguments), against the L-PDFT eigenvalues in <out>/lpdft.json.

    python nevpt2_reference.py --rings 4 --out tc_l1.2_s0.8 --slip-long 1.2 --slip-short 0.8 --nroots 9

Separates the error of the on-top functional from that of the small active space: NEVPT2 adds the dynamic correlation
of the doubly occupied and virtual orbitals to the same CASCI states.
"""
import argparse, json, os, sys, time
import numpy as np
from pyscf import mcscf, mrpt, fci

here = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--nroots', type=int, default=9)
p.add_argument('--nfront', type=int, default=2)
ref, rest = p.parse_known_args()   # (not args: the exec below defines its own)
# rebuild the system exactly as acene_dimer.py does (geometry, RHF from the checkpoint, clusters)
sys.argv = [os.path.join(here, 'acene_dimer.py'), 'analyze', '--out', ref.out, '--nfront', str(ref.nfront)] + rest
src = open(os.path.join(here, 'acene_dimer.py')).read().split("orbfile = ")[0]
exec(compile(src, 'acene_dimer.py', 'exec'))
cmf.set_mo_coeffs(np.load(os.path.join(ref.out, 'cmf_orbitals.npy')))
C = np.hstack((cmf.Cfrz, cmf.Cdoc, cmf.Cact, cmf.Cvir))
n, k = cmf.Cact.shape[1], ref.nfront
EV = 27.211386245988

mc = mcscf.CASCI(mf, n, (2 * k, 2 * k))
mc.fcisolver = fci.direct_spin1.FCI(mol)
mc.fcisolver.nroots = ref.nroots
mc.fcisolver.conv_tol = 1e-12
mc.fix_spin_(ss=0)
mc.canonicalization = False
mc.verbose = 0
mc.kernel(C)
e_cas = np.array(mc.e_tot)
e_pt = []
for i in range(ref.nroots):
    t0 = time.time()
    e2 = mrpt.NEVPT(mc, root=i).kernel()
    e_pt.append(e_cas[i] + e2)
    print('root %d  CASCI %.8f  SC-NEVPT2 %.8f  (%.0f s)' % (i, e_cas[i], e_pt[-1], time.time() - t0), flush=True)
e_pt = np.array(e_pt)
out = {'casci': e_cas.tolist(), 'nevpt2': e_pt.tolist()}
lp = os.path.join(ref.out, 'lpdft.json')
if os.path.exists(lp):
    L = json.load(open(lp))
    for x in [k for k in L if k not in ('tpsci_singlets', 'diabatic')]:
        out['lpdft_' + x] = L[x]['e']
print('excitation energies (eV):')
print('  CASCI     ', np.round((e_cas - e_cas[0]) * EV, 3))
print('  SC-NEVPT2 ', np.round((np.sort(e_pt) - np.sort(e_pt)[0]) * EV, 3))
for x in [k for k in out if k.startswith('lpdft_')]:
    e = np.array(out[x])
    print('  %-10s' % x[6:], np.round((e - e[0]) * EV, 3))
json.dump(out, open(os.path.join(ref.out, 'nevpt2.json'), 'w'), indent=1)
