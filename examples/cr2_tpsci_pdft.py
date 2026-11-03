"""
PDFT on the TPSCI states of [Cr2(OH)3(tmtacn)2]3+ from the 2024 Faraday Discussions data
(github.com/arnab82/2024_faraday_discussions_data, directory cr2): (32e, 38o) active space of Cr 3d4d and O 2p3p
fragment orbitals (SVD partitioning of the def2-SVP ROHF orbitals, `get_orbitals_ints.py`), optimized by cMF inside
the active space (`cmf.jl`, Ccmf.npy), the 116 other doubly occupied ROHF orbitals frozen; TPSCI with M = 100
cluster states (`m100/tpsci_05.jl`).

    julia --project=/path/to/TPSChem.jl -t auto tpsci_rdms_from_jld2.jl DATA/cr2/data_cmf.jld2 \\
          DATA/cr2/m100/tpsci_05.jl.scr/data_tpsci_04.jld2 tpsci_rdms_04.npz
    python cr2_tpsci_pdft.py DATA/cr2 tpsci_rdms_04.npz --functionals tPBE,ftPBE,MC23

Without the RDM file it only rebuilds the active-space problem and checks it against the integrals of the data
(ints_h0.npy, ints_h1.npy), which validates the frozen doubly occupied orbitals. J is for H = -2J S_A.S_B (the
experimental -66 cm-1 is in this convention, see the README of the data).
"""
import argparse, json, os
import numpy as np
from pyscf import gto, scf, lib
import pycmf
from pycmf import pdft

CM = 219474.6313705
p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('data', help='the cr2 directory of 2024_faraday_discussions_data')
p.add_argument('rdms', nargs='?', help='TPSCI RDMs from tpsci_rdms_from_jld2.jl (npz)')
p.add_argument('--functionals', default='tLDA,ftLDA,tPBE,ftPBE,tBLYP,ftBLYP,trevPBE,ftrevPBE,tOPBE,tM06L,tPBE0,MC23')
p.add_argument('--grids-level', type=int, default=3)
p.add_argument('--auxbasis', default='def2-universal-jkfit')
p.add_argument('--rohf-chk', default='rohf_faraday.chk', help='ROHF checkpoint (computed if missing)')
p.add_argument('--max-memory', type=int, default=10000)
args = p.parse_args()

# ---------------------------------------------------------------- ROHF (high spin), as get_orbitals_ints.py
mol = gto.M(atom=os.path.join(args.data, 'geometry.xyz'), basis='def2-svp', charge=3, spin=6,
            max_memory=args.max_memory, verbose=0)
mf = scf.ROHF(mol).density_fit(auxbasis=args.auxbasis)
if os.path.exists(args.rohf_chk):
    mf.__dict__.update(lib.chkfile.load(args.rohf_chk, 'scf'))
else:
    mf.chkfile = args.rohf_chk
    mf.conv_tol = 1e-9
    mf.kernel()
print('ROHF %.10f' % mf.e_tot)

# ---------------------------------------------------------------- orbital spaces of the data
S = mf.get_ovlp()
Cact0 = np.load(os.path.join(args.data, 'mo_coeffs.npy'))      # fragment orbitals before cMF
Ccmf = np.load(os.path.join(args.data, 'Ccmf.npy'))            # after cMF (rotation within the active space)
nact = Cact0.shape[1]
# the frozen doubly occupied orbitals: the ROHF doubly occupied space without its part in the active space
Q = np.eye(mol.nao) - Cact0 @ Cact0.T @ S
Cdocc = mf.mo_coeff[:, mf.mo_occ == 2]
w, u = np.linalg.eigh(Cdocc.T @ S @ Q @ Cdocc)
keep = w > .5
Cenv = Q @ Cdocc @ u[:, keep] / np.sqrt(w[keep])
print('frozen doubly occupied orbitals: %d (projected norms %.1e .. %.6f; dropped %d below %.1e)'
      % (keep.sum(), w[keep].min(), w[keep].max(), (~keep).sum(), w[~keep].max()))
assert 2*Cenv.shape[1] + 32 == mol.nelectron
# the remaining (virtual) orbitals, for completeness
Cocc = np.hstack((Cenv, Ccmf))
w, u = np.linalg.eigh(S - S @ Cocc @ Cocc.T @ S)
Cvir = np.linalg.solve(S, S @ u[:, w > .5]) / np.sqrt(w[w > .5])

clusters = [list(range(0, 10)), list(range(10, 20)), list(range(20, 26)), list(range(26, 32)), list(range(32, 38))]
fspace = [(5, 2), (2, 5), (3, 3), (3, 3), (3, 3)]
cmf = pycmf.CMF(mf)
cmf.set_Cdoc(Cenv)
cmf.set_Cact(Ccmf, clusters, fspace)
cmf.set_Cvir(Cvir)

# check against the integrals of the data (active orbitals before cMF; density-fitted J/K here, exact there)
vcore = cmf.get_core_veff()
h0_data, h1_data = float(np.load(os.path.join(args.data, 'ints_h0.npy'))), np.load(os.path.join(args.data, 'ints_h1.npy'))
D = 2*Cenv @ Cenv.T
h0 = mf.energy_nuc() + np.einsum('ij,ji->', D, mf.get_hcore() + .5*vcore)
h1 = Cact0.T @ (mf.get_hcore() + vcore) @ Cact0
print('check against the data: h0 %.8f vs %.8f (%.1e), max |h1 - h1_data| = %.1e'
      % (h0, h0_data, h0 - h0_data, abs(h1 - h1_data).max()))

if args.rdms is None:
    raise SystemExit
# ---------------------------------------------------------------- TPSCI RDMs and PDFT
e_ci, dm1s, dm2s = pdft.load_tpsci_rdms(args.rdms)
h0_a, h1_a, h2_a, orbs = cmf.get_active_hamiltonian()
assert list(orbs) == list(range(nact))
rdms, e_by_s, out = {}, {}, {'e_tpsci': list(e_ci)}
for r in range(len(e_ci)):
    ss = pdft.spin_square_rdm(dm1s[r], dm2s[r])
    s = round(np.sqrt(1 + 4*max(ss, 0)) - 1) / 2
    e_rdm = h0_a + np.einsum('pq,pq', h1_a, dm1s[r]) + .5*np.einsum('pqrs,pqrs', h2_a, dm2s[r])
    print('root %d  E(TPSCI) %.8f  E from RDMs %.8f (%.1e)  N %.6f  <S^2> %.4f  S %.1f'
          % (r, e_ci[r], e_rdm, e_rdm - e_ci[r], np.trace(dm1s[r]), ss, s))
    if s in rdms:
        print('  (a second root with S = %.1f, the lowest one is kept)' % s)
        continue
    rdms[s], e_by_s[s] = (dm1s[r], dm2s[r]), e_ci[r]

spins = np.array(sorted(rdms))
A = np.vstack((np.ones_like(spins), -spins*(spins + 1))).T
out['j_tpsci'] = np.linalg.lstsq(A, np.array([e_by_s[x] for x in spins]), rcond=None)[0][1]*CM
print('TPSCI (variational) J = %.2f cm-1' % out['j_tpsci'])

functionals = args.functionals.split(',')
scan = pdft.functional_scan(cmf, rdms, functionals, grids_level=args.grids_level)
out['j_wfn_rdm'] = scan['wfn'][1]*CM
print('J of the RDM energies (density-fitted integrals): %.2f cm-1' % out['j_wfn_rdm'])
for x in functionals:
    out['j_tpsci_pdft_' + x] = scan[x][1]*CM
    print('TPSCI-PDFT %-9s J = %8.2f cm-1' % (x, scan[x][1]*CM))
json.dump(out, open(os.path.splitext(args.rdms)[0] + '_pdft.json', 'w'), indent=1, default=float)
