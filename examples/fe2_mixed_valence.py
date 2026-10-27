"""
cMF for the Fe2(mu-H)2 complex in arnab82/Mixed_Valence_bimetallics (Fe2_mixed_jeremy), then export for TPSCI.

The UHF solution there is broken symmetry and spin contaminated (<S^2> = 12.6 for M_s = 0) with
alternating spin density on the ligand carbons, so localizing its orbitals onto the two Fe is ill defined.
Here the clusters are the orthogonalized 3d AOs of each Fe, and the UHF density is only used to split
the other orbitals into doubly occupied and virtual ones. The charge and spin of each Fe come from
`fspace`. The orbitals are optimized for the average of a valence-trapped configuration and its mirror
image, so neither is favored, and the result is written for FermiCG.jl.

Meta-Lowdin 3d populations of the UHF:  Fe0 n = 6.11 (5.0 alpha, 1.1 beta),  Fe1 n = 5.45 (0.5 alpha, 5.0 beta)

All other orbitals are doubly occupied or empty, so the 3d electron count has to be even here
(436 electrons). For an odd count, add a cluster for the bridging hydrides, e.g. ['2 H 1s', '3 H 1s'].

Cost: 1116 AOs. Each orbital step is two J/K builds plus two passes over the density fitting integrals
(about 25 GB with def2-universal-jkfit).
"""
import sys
from pyscf import lib, scf
from pyscf.tools import molden
import pycmf

chkfile = sys.argv[1] if len(sys.argv) > 1 else 'Fe2_mixed_jeremy/mf.chk'

mol = lib.chkfile.load_mol(chkfile)
mol.verbose = 4
mol.max_memory = 400000

# Same Hamiltonian as the UHF (scalar X2C), with density fitting for J/K and the cluster integrals
mf = scf.sfx2c(scf.UHF(mol)).density_fit(auxbasis='def2-universal-jkfit')
mf.__dict__.update(scf.chkfile.load(chkfile, 'scf'))

# Fe(I) Fe(III) and its mirror image, sharing one set of orbitals
# (for Fe(II) Fe(II) use a single fspace, e.g. [(5, 1), (1, 5)], without state_average_)
fspaces = [[(5, 2), (0, 5)],
           [(5, 0), (2, 5)]]

cmf = pycmf.CMF(mf)
cmf.set_clusters_from_ao([['0 Fe 3d'], ['1 Fe 3d']], fspaces[0])
cmf.state_average_(fspaces)
cmf.conv_tol_grad = 1e-4
cmf.kernel()

for k in range(len(fspaces)):
    cmf.select_state(k)
    cmf.analyze()

cmf.select_state(0)
cmf.export_fermicg('fe2_cmf.npz')
molden.from_mo(mol, 'fe2_cmf.molden', cmf.mo_coeff)
