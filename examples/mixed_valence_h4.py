"""
Toy mixed-valence system: (H2)2+, i.e. H2 + H2+ at separation R.

ROHF delocalizes the hole over both molecules, which is wrong at large R. In cMF the charge and spin of
each cluster are set by `fspace`, and the cluster orbitals come straight from the atomic orbitals of each
molecule (`set_clusters_from_ao`), so no mean-field orbitals have to be localized. The two charge-localized
states, H2 + H2+ and H2+ + H2, are degenerate. Their coupling (hole hopping) is intercluster correlation,
which is left to post-cMF methods.
"""
from pyscf import gto, scf, fci
import pycmf

for R in [2.0, 3.0, 5.0, 10.0]:
    mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0, 0, .74)], ['H', (R, 0, 0)], ['H', (R, 0, .74)]],
                basis='6-31g', charge=1, spin=1, verbose=0)
    mf = scf.ROHF(mol).run()
    e_fci = fci.FCI(mf).kernel()[0]

    cmf = pycmf.CMF(mf)
    cmf.set_clusters_from_ao([['0 H', '1 H'], ['2 H', '3 H']], [(1, 1), (1, 0)])
    e_cmf = cmf.kernel()
    print(" R = %5.1f   ROHF %14.8f   cMF %14.8f   FCI %14.8f" % (R, mf.e_tot, e_cmf, e_fci))

# Properties of the last cMF state: spin densities, populations and dipole moment
cmf.verbose = 4
cmf.analyze()
dm_a, dm_b = cmf.make_rdm1s(ao_repr=True)   # AO densities, e.g. for pyscf.prop or cubegen

# Orbitals shared by both charge-localized states, and export for TPSCI in FermiCG.jl
cmf = pycmf.CMF(mf)
cmf.set_clusters_from_ao([['0 H', '1 H'], ['2 H', '3 H']], [(1, 1), (1, 0)])
cmf.state_average_([[(1, 1), (1, 0)], [(1, 0), (1, 1)]])
cmf.kernel()
print(" state-averaged cMF: E = %14.8f, states %s" % (cmf.e_tot, " ".join("%.8f" % e for e in cmf.e_states)))
cmf.export_fermicg('h4_cation_cmf.npz')
