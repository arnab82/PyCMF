"""
cMF for the Fe2(mu-H)2 complex in arnab82/Mixed_Valence_bimetallics (Fe2_mixed_jeremy).

The UHF solution there is broken symmetry and spin contaminated (<S^2> = 12.6 for M_s = 0) with
alternating spin density on the ligand carbons, so localizing its orbitals onto the two Fe is ill defined.
Here the clusters are the orthogonalized 3d AOs of each Fe, and the UHF density is only used to split
the other orbitals into doubly occupied and virtual ones. The charge and spin of each Fe come from
`fspace`, so different valence-trapped configurations can be compared directly.

Meta-Lowdin 3d populations of the UHF:  Fe0 n = 6.11 (5.0 alpha, 1.1 beta),  Fe1 n = 5.45 (0.5 alpha, 5.0 beta)

Cost: 1116 AOs. Each energy evaluation needs a few J/K builds plus one integral pass per cluster, so use
density fitting (about 25 GB for the 3-index integrals with def2-universal-jkfit) and `sequential = False`,
which builds the J/K of all clusters in a single pass.
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

# Valence-trapped configurations to compare (the 3d electron count has to be even here,
# since all other orbitals are doubly occupied or empty)
configurations = {
    'Fe(II) Fe(II)':  [(5, 1), (1, 5)],
    'Fe(I) Fe(III)':  [(5, 2), (0, 5)],
    'Fe(III) Fe(I)':  [(5, 0), (2, 5)],
}

for name, fspace in configurations.items():
    cmf = pycmf.CMF(mf)
    cmf.set_clusters_from_ao([['0 Fe 3d'], ['1 Fe 3d']], fspace)
    cmf.sequential = False
    cmf.conv_tol_grad = 1e-4
    e = cmf.kernel()
    print(" %-16s  E = %16.10f  converged = %s" % (name, e, cmf.converged))
    cmf.analyze()
    molden.from_mo(mol, 'cmf_%s.molden' % name.replace(' ', '_').replace('(', '').replace(')', ''), cmf.mo_coeff)
