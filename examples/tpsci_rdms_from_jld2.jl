# RDMs of a saved TPSCI state (jld2 from FermiCG or TPSChem) for PDFT in PyCMF (examples/cr2_tpsci_pdft.py,
# pycmf.pdft.load_tpsci_rdms), with TPSChem.jl
#
#   julia --project=/path/to/TPSChem.jl -t auto tpsci_rdms_from_jld2.jl data_cmf.jld2 data_tpsci_04.jld2 out.npz
#
# data_cmf.jld2 has clusters, init_fspace, ints, d1 (cmf.jl); data_tpsci_XX.jld2 has the TPSCI state v0a and its
# energies e0a (and PT2 corrections e2a). The cluster bases are not saved with the state, so they are rebuilt as in
# the TPSCI run: TPSCI_M cluster states per Fock sector (default 100) within TPSCI_DELTA electrons of the reference
# (default 3), from compute_cluster_eigenbasis_spin. The energy of each root from its RDMs is compared with e0a, which
# checks that the rebuilt bases are the ones the state was computed in (signs and order of the cluster states).
# Output, as examples/tpsci_rdms.jl: energies (R,), spin-summed 1RDMs dm1 (n, n, R) and spin-free 2RDMs
# dm2 (n, n, n, n, R) with G[p,q,r,s] = <p'q'sr>, in the orbital order of the clusters; e2a when present.
using TPSChem
using TPSChem.QCBase
using TPSChem.RDM
using TPSChem.InCoreIntegrals
using JLD2
using NPZ
using Printf
using LinearAlgebra

cmffile, tpscifile, outfile = ARGS[1], ARGS[2], ARGS[3]
M = parse(Int, get(ENV, "TPSCI_M", "100"))
delta = parse(Int, get(ENV, "TPSCI_DELTA", "3"))

# files written by FermiCG (2024) refer to its types and to the then separate QCBase, RDM and InCoreIntegrals packages
typemap = Dict("FermiCG.TPSCIstate" => TPSChem.TPSCIstate,
               "FermiCG.FockConfig" => TPSChem.FockConfig,
               "FermiCG.ClusterConfig" => TPSChem.ClusterConfig,
               "QCBase.MOCluster" => TPSChem.QCBase.MOCluster,
               "InCoreIntegrals.InCoreInts" => TPSChem.InCoreIntegrals.InCoreInts,
               "RDM.RDM1" => TPSChem.RDM.RDM1)
cmf = JLD2.load(cmffile; typemap=typemap)
tps = JLD2.load(tpscifile; typemap=typemap)
ints, clusters, d1 = cmf["ints"], cmf["clusters"], cmf["d1"]
init_fspace = FockConfig(cmf["init_fspace"])
v0a, e0a = tps["v0a"], tps["e0a"]
n, R = n_orb(ints), length(e0a)
@printf(" %d clusters, %d orbitals, reference %s, %d roots, %d configurations\n",
        length(clusters), n, init_fspace, R, length(v0a))

cluster_bases = TPSChem.compute_cluster_eigenbasis_spin(ints, clusters, d1, [delta for c in clusters], init_fspace,
                                                        max_roots=M, verbose=1)
cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints)
TPSChem.add_spinfree_2rdm_ops!(cluster_ops, cluster_bases)

γ_aa, γ_bb = TPSChem.compute_1rdm(v0a, cluster_ops)
Γ = TPSChem.compute_2rdm(v0a, cluster_ops)

dm1 = zeros(n, n, R)
dm2 = zeros(n, n, n, n, R)
maxdiff = 0.0
for r in 1:R
    dm1[:, :, r] .= γ_aa[:, :, r, r] .+ γ_bb[:, :, r, r]
    dm2[:, :, :, :, r] .= Γ[:, :, :, :, r, r]
    # E = h0 + sum h1 D1 + 1/2 sum (pq|rs) <p'r'sq>, with <p'r'sq> = G[p,r,q,s]
    e = ints.h0 + sum(ints.h1 .* dm1[:, :, r]) + 0.5 * sum(ints.h2 .* permutedims(dm2[:, :, :, :, r], (1, 3, 2, 4)))
    global maxdiff = max(maxdiff, abs(e - e0a[r]))
    @printf(" root %2d  E(TPSCI) = %.10f  E(RDMs) = %.10f  diff %.1e  Tr(dm1) = %.6f\n",
            r, e0a[r], e, e - e0a[r], tr(dm1[:, :, r]))
end
maxdiff < 1e-6 || @warn @sprintf("the RDM energies differ from e0a by up to %.1e: the rebuilt cluster bases are not those of the TPSCI run", maxdiff)

out = Dict("energies" => collect(e0a), "dm1" => dm1, "dm2" => dm2)
haskey(tps, "e2a") && (out["e2a"] = collect(tps["e2a"]))
npzwrite(outfile, out)
@printf(" wrote %s\n", outfile)
