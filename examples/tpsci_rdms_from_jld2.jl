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
#
# The bases are trimmed to the Fock sectors and cluster states that the TPSCI state uses (which leaves every matrix
# element unchanged). The 2RDMs of each root (no transition RDMs) come from tpsci_rdm_tools.jl: the blocks with all
# four indices on one cluster from natural cluster states (exact, without TPSChem's Ppqsr tables of n^4 m^2 numbers
# per sector, 6.6 GB per 10-orbital cluster here), the other blocks from TPSChem's inter-cluster loop, threaded over
# cluster combinations with a progress line each. TPSCI_ONCLUSTER=table uses Ppqsr and compute_2rdm instead.
# Output: first <out>_1rdm.npz (dm1a, dm1b, dm1, natural occupations occ, Sz), written before the 2RDM step; then
# <out>.npz as examples/tpsci_rdms.jl: energies (R,) including h0, spin-summed 1RDMs dm1 (n, n, R) and spin-free
# 2RDMs dm2 (n, n, n, n, R) with G[p,q,r,s] = <p'q'sr>, in the orbital order of the clusters; e2a when present.
using TPSChem
using TPSChem.QCBase
using TPSChem.RDM
using TPSChem.InCoreIntegrals
using JLD2
using NPZ
using Printf
using LinearAlgebra
using StaticArrays
include(joinpath(@__DIR__, "tpsci_rdm_tools.jl"))

cmffile, tpscifile, outfile = ARGS[1], ARGS[2], ARGS[3]
M = parse(Int, get(ENV, "TPSCI_M", "100"))
delta = parse(Int, get(ENV, "TPSCI_DELTA", "3"))
oncluster = get(ENV, "TPSCI_ONCLUSTER", "natural")

# files written by FermiCG (2024) refer to its types and to the then separate QCBase, RDM and InCoreIntegrals packages:
# map them to TPSChem's (same layout), with the stored type parameters, and leave everything else to JLD2
const OLD_TYPES = Dict("FermiCG.TPSCIstate" => TPSChem.TPSCIstate,
                       "FermiCG.FockConfig" => TPSChem.FockConfig,
                       "FermiCG.ClusterConfig" => TPSChem.ClusterConfig,
                       "QCBase.MOCluster" => TPSChem.QCBase.MOCluster,
                       "InCoreIntegrals.InCoreInts" => TPSChem.InCoreIntegrals.InCoreInts,
                       "RDM.RDM1" => TPSChem.RDM.RDM1)
old_types(f, path, params) = haskey(OLD_TYPES, path) ?
    (isempty(params) ? OLD_TYPES[path] : OLD_TYPES[path]{params...}) : JLD2.default_typemap(f, path, params)
read_all(fname) = jldopen(fname, "r"; typemap=old_types) do f
    Dict(k => f[k] for k in keys(f))
end
cmf = read_all(cmffile)
tps = read_all(tpscifile)
ints, clusters, d1 = cmf["ints"], cmf["clusters"], cmf["d1"]
init_fspace = FockConfig(cmf["init_fspace"])
v0a, e0a = tps["v0a"], tps["e0a"]
n, R = n_orb(ints), length(e0a)
@printf(" %d clusters, %d orbitals, reference %s, %d roots, %d configurations\n",
        length(clusters), n, init_fspace, R, length(v0a))

cluster_bases = TPSChem.compute_cluster_eigenbasis_spin(ints, clusters, d1, [delta for c in clusters], init_fspace,
                                                        max_roots=M, verbose=1)

# keep the Fock sectors and the leading cluster states that the state uses
used = [Dict{Tuple{Int16,Int16},Int}() for c in clusters]
for (fock, configs) in v0a.data, (config, coeff) in configs, i in eachindex(clusters)
    used[i][fock[i]] = max(get(used[i], fock[i], 0), Int(config[i]))
end
for (i, cb) in enumerate(cluster_bases)
    for sector in collect(keys(cb.basis))
        if haskey(used[i], sector)
            sol, m = cb.basis[sector], used[i][sector]
            cb.basis[sector] = TPSChem.Solution(sol.ansatz, sol.energies[1:m], sol.vectors[:, 1:m])
        else
            delete!(cb.basis, sector)
        end
    end
    @printf(" cluster %d: %d Fock sectors, %d cluster states used\n", i, length(cb.basis), sum(values(used[i])))
end

cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints)
offsets = cumsum([0; [length(c) for c in clusters[1:end-1]]])

# 1RDMs first, written before the (long) 2RDM step: <outfile>_1rdm.npz with dm1a, dm1b, dm1 (n, n, R), the natural
# occupations occ (n, R) and Sz (R,). <S^2> needs the 2RDM: it comes with the second file
out1 = splitext(outfile)[1] * "_1rdm.npz"
if isfile(out1)     # from an earlier run on the same state (TPSChem's compute_1rdm takes ~15 min here)
    d = npzread(out1)
    dm1a, dm1b = d["dm1a"], d["dm1b"]
    maximum(abs.(d["energies"] .- e0a)) < 1e-10 || error("$out1 is for another state")
    @printf(" 1RDMs read from %s\n", out1)
else
    @time γ_aa, γ_bb = TPSChem.compute_1rdm(v0a, cluster_ops)
    dm1a = cat([γ_aa[:, :, r, r] for r in 1:R]..., dims=3)
    dm1b = cat([γ_bb[:, :, r, r] for r in 1:R]..., dims=3)
end
dm1 = dm1a .+ dm1b
occ = hcat([reverse(eigvals(Symmetric(dm1[:, :, r]))) for r in 1:R]...)
sz = [0.5 * (tr(dm1a[:, :, r]) - tr(dm1b[:, :, r])) for r in 1:R]
for r in 1:R
    @printf(" root %2d  N = %.6f  Sz = %.4f  natural occupations %s\n", r, tr(dm1[:, :, r]), sz[r],
            join([@sprintf("%.3f", x) for x in occ[:, r] if 0.02 < x < 1.98], " "))
end
npzwrite(out1, Dict("energies" => collect(e0a), "dm1a" => dm1a, "dm1b" => dm1b, "dm1" => dm1, "occ" => occ, "sz" => sz))
@printf(" wrote %s\n", out1)
flush(stdout)

if oncluster == "table"
    TPSChem.add_spinfree_2rdm_ops!(cluster_ops, cluster_bases)
    Γ = TPSChem.compute_2rdm(v0a, cluster_ops)
    dm2 = cat([Γ[:, :, :, :, r, r] for r in 1:R]..., dims=5)
else
    # root-diagonal blocks only: the roots are the spin states S = 0..3, between which the spin-free transition
    # RDMs vanish, and PDFT needs the RDM of each state
    @time dm2 = intercluster_2rdm(v0a, cluster_ops, offsets)
    @time oncluster_2rdm!(dm2, v0a, cluster_bases, offsets)
end

# E = h0 + sum h1 D1 + 1/2 sum (pq|rs) <p'r'sq>, with <p'r'sq> = G[p,r,q,s]. TPSChem's TPSCI energies leave out h0,
# FermiCG's may include it: the check accepts either and says which
e_rdm = [ints.h0 + sum(ints.h1 .* dm1[:, :, r]) + 0.5 * sum(ints.h2 .* permutedims(dm2[:, :, :, :, r], (1, 3, 2, 4)))
         for r in 1:R]
d_with, d_without = maximum(abs.(e_rdm .- e0a)), maximum(abs.(e_rdm .- ints.h0 .- e0a))
shift = d_with <= d_without ? 0.0 : ints.h0
@printf(" e0a %s h0 = %.8f\n", shift == 0 ? "includes" : "leaves out", ints.h0)
for r in 1:R
    @printf(" root %2d  E(TPSCI) = %.10f  E(RDMs) = %.10f  diff %.1e  Tr(dm1) = %.6f\n",
            r, e0a[r] + shift, e_rdm[r], e_rdm[r] - e0a[r] - shift, tr(dm1[:, :, r]))
end
maxdiff = min(d_with, d_without)
maxdiff < 1e-6 || @warn @sprintf("the RDM energies differ from e0a by up to %.1e: the rebuilt cluster bases are not those of the TPSCI run", maxdiff)

out = Dict("energies" => collect(e0a) .+ shift, "dm1" => dm1, "dm2" => dm2)
haskey(tps, "e2a") && (out["e2a"] = collect(tps["e2a"]))
npzwrite(outfile, out)
@printf(" wrote %s\n", outfile)
