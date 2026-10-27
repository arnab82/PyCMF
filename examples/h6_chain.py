"""
Linear H6/STO-3G, three clusters of two Lowdin orbitals.

Same system as test/test_cmf.jl in ClusterMeanField.jl:
    CMF-CI (Lowdin orbitals)  -2.972938136549
    ooCMF                     -3.205983033016
"""
from pyscf import gto, scf
import pycmf

mol = gto.M(atom=[['H', (i, 0, 0)] for i in range(6)], basis='sto-3g', verbose=4)
mf = scf.RHF(mol).run()

cmf = pycmf.CMF(mf)
cmf.set_Cact(cmf.lowdin(), [[0, 1], [2, 3], [4, 5]], [(1, 1), (1, 1), (1, 1)])

e_ci = cmf.cmf_ci()     # cluster states in the fixed Lowdin orbitals
e_oo = cmf.kernel()     # orbitals and cluster states optimized
print(" CMF-CI: %16.12f" % e_ci)
print(" ooCMF:  %16.12f" % e_oo)
cmf.analyze()
