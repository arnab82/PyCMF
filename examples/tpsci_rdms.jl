# TPSCI states and their RDMs for PDFT (pycmf.pdft.tpsci_spin_ladder), with TPSChem.jl
#
#   python:  cmf.export_fermicg('cmf.npz')                       # all clusters (or a subset, see below)
#   julia:   julia --project=/path/to/TPSChem.jl -t auto tpsci_rdms.jl cmf.npz tpsci_rdms.npz
#   python:  states, j_tpsci, j_pdft = pycmf.pdft.tpsci_spin_ladder(cmf, 'tpsci_rdms.npz', otxc='tPBE')
#
# Settings (environment variables): TPSCI_NROOTS (default 4, e.g. 2*min(S_A, S_B) + 1 for a spin ladder),
# TPSCI_M (cluster states per Fock sector, 100), TPSCI_DELTA (charge range of the cluster bases, 2),
# TPSCI_THRESH_CIPSI (1e-3), TPSCI_THRESH_FOI (1e-6).
# The output has the energies (R,), the spin-summed 1RDMs dm1 (n, n, R) and the spin-free 2RDMs
# dm2 (n, n, n, n, R) with TPSChem's convention G[p,q,r,s] = <p'q'sr>, in the orbital order of the export.
using TPSChem
using TPSChem.QCBase
using TPSChem.InCoreIntegrals
using TPSChem.RDM
using NPZ
using Printf
using LinearAlgebra

infile, outfile = ARGS[1], ARGS[2]
nroots = parse(Int, get(ENV, "TPSCI_NROOTS", "4"))
M = parse(Int, get(ENV, "TPSCI_M", "100"))
delta = parse(Int, get(ENV, "TPSCI_DELTA", "2"))
thresh_cipsi = parse(Float64, get(ENV, "TPSCI_THRESH_CIPSI", "1e-3"))
thresh_foi = parse(Float64, get(ENV, "TPSCI_THRESH_FOI", "1e-6"))

d = npzread(infile)
ints = InCoreInts(first(d["h0"]), d["h1"], d["h2"])
nclus = maximum(d["cluster_index"])
clusters = [MOCluster(i, findall(==(i), d["cluster_index"])) for i in 1:nclus]
init_fspace = FockConfig([Tuple(d["fspace"][i, :]) for i in 1:nclus])
d1 = RDM1(d["rdm1a"], d["rdm1b"])
@printf(" %d clusters, %d orbitals, reference %s, cMF energy %.10f\n", nclus, n_orb(ints), init_fspace, first(d["e_cmf"]))

cluster_bases = TPSChem.compute_cluster_eigenbasis_spin(ints, clusters, d1, [delta for i in 1:nclus], init_fspace,
                                                        max_roots=M, verbose=1)
clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
cluster_ops = TPSChem.compute_cluster_ops_2rdm(cluster_bases, ints)
TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

ci_vector = TPSChem.TPSCIstate(clusters, init_fspace, R=nroots)
ci_vector = TPSChem.add_spin_focksectors(ci_vector)
TPSChem.eye!(ci_vector)
e0, v0 = TPSChem.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=true,
                          thresh_cipsi=thresh_cipsi, thresh_foi=thresh_foi, thresh_asci=-1)

γ_aa, γ_bb = TPSChem.compute_1rdm(v0, cluster_ops)
Γ = TPSChem.compute_2rdm(v0, cluster_ops)
n = n_orb(ints)
dm1 = zeros(n, n, nroots)
dm2 = zeros(n, n, n, n, nroots)
for r in 1:nroots
    dm1[:, :, r] .= γ_aa[:, :, r, r] .+ γ_bb[:, :, r, r]
    dm2[:, :, :, :, r] .= Γ[:, :, :, :, r, r]
    @printf(" root %2d  E = %.10f  Tr(dm1) = %.6f\n", r, e0[r], tr(dm1[:, :, r]))
end
npzwrite(outfile, Dict("energies" => collect(e0), "dm1" => dm1, "dm2" => dm2))
@printf(" wrote %s\n", outfile)
