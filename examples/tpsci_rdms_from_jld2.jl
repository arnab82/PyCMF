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
# element unchanged). The 2RDM blocks with all four indices on one cluster are computed from that cluster's density
# matrix in its state basis, rho_uv per sector, diagonalized into natural cluster states phi_k with weights w_k:
# sum_uv rho_uv <u|p'q'sr|v> = sum_k w_k <phi_k|p'q'sr|phi_k>, an FCI 2RDM per natural state. This is exact and avoids
# TPSChem's Ppqsr tables (n^4 m^2 numbers per sector, 6.6 GB per 10-orbital cluster here); the other blocks come
# from TPSChem's assembler. TPSCI_ONCLUSTER=table uses Ppqsr and compute_2rdm instead.
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
using StaticArrays

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

"""
On-cluster 2RDM blocks (all four indices on one cluster) of each root, from natural cluster states
"""
function oncluster_2rdm!(Γ, v::TPSChem.TPSCIstate{T,N,R}, bases, offsets; wmin=1e-14) where {T,N,R}
    for ci in v.clusters
        I, n, off = ci.idx, length(ci), offsets[ci.idx]
        rho = Dict{Tuple{Int16,Int16},Array{T,3}}()
        for (fock, configs) in v.data
            m = size(bases[I][fock[I]].vectors, 2)
            ρ = get!(() -> zeros(T, m, m, R), rho, fock[I])
            groups = Dict{Vector{Int16},Vector{Pair{Int16,MVector{R,T}}}}()
            for (config, coeff) in configs
                key = [config[k] for k in 1:N if k != I]
                push!(get!(groups, key, Pair{Int16,MVector{R,T}}[]), config[I] => coeff)
            end
            for list in values(groups), (u, cu) in list, (w, cw) in list, r in 1:R
                ρ[u, w, r] += cu[r] * cw[r]
            end
        end
        for (sector, ρ) in rho
            sol = bases[I][sector]
            for r in 1:R
                F = eigen(Symmetric(ρ[:, :, r]))
                for k in eachindex(F.values)
                    wk = F.values[k]
                    abs(wk) < wmin && continue
                    φ = sol.vectors * F.vectors[:, k]
                    _, _, aa, bb, ab = TPSChem.ActiveSpaceSolvers.FCI.compute_rdm1_rdm2(sol.ansatz, φ, φ)
                    for s in 1:n, rr in 1:n, q in 1:n, p in 1:n
                        Γ[off+p, off+q, off+rr, off+s, r, r] += wk * (aa[p, rr, q, s] + bb[p, rr, q, s] +
                                                                     ab[p, rr, q, s] + ab[q, s, p, rr])
                    end
                end
            end
        end
    end
    return Γ
end

cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints)
γ_aa, γ_bb = TPSChem.compute_1rdm(v0a, cluster_ops)
offsets = cumsum([0; [length(c) for c in clusters[1:end-1]]])
if oncluster == "table"
    TPSChem.add_spinfree_2rdm_ops!(cluster_ops, cluster_bases)
    Γ = TPSChem.compute_2rdm(v0a, cluster_ops)
else
    Γ = zeros(n, n, n, n, R, R)
    @time TPSChem._accumulate_2rdm_intercluster_full!(Γ, v0a, v0a, cluster_ops, offsets)
    @time oncluster_2rdm!(Γ, v0a, cluster_bases, offsets)
end

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
