# TPSCI states with their RDMs and the transition RDMs between states of the same spin, for multistate PDFT
# (pycmf.lpdft) and diabatic analysis of singlet fission.
#
#   julia --project=/path/to/TPSChem.jl -t auto tpsci_states.jl cmf_export.npz states.npz
#
# cmf_export.npz from CMF.export_fermicg. Settings: TPSCI_NROOTS (8), TPSCI_M (100), TPSCI_DELTA (2),
# TPSCI_THRESH_CIPSI (1e-4), TPSCI_THRESH_FOI (1e-6), TPSCI_CONV (1e-7), TPSCI_CI_CONV (1e-9), TPSCI_SPIN_TOL (0.1: |<S^2>_I - <S^2>_J| for a same-spin pair).
# Output: energies (R,) with h0, s2 (R,), pairs (P, 2) 1-based root pairs (I <= J; I = J are the state RDMs), and per
# channel dm1 (n, n, P) and dm2 (n, n, n, n, P), spin-free, TPSChem convention G[p,q,r,s] = <p'q'sr>, transition
# channels symmetrized ((G_IJ + G_JI) / 2), in the orbital order of the export; cluster_bases and the state go to
# <states>.jld2.
using TPSChem
using TPSChem.QCBase
using TPSChem.InCoreIntegrals
using TPSChem.RDM
using NPZ
using JLD2
using Printf
using LinearAlgebra
using StaticArrays
include(joinpath(@__DIR__, "..", "tpsci_rdm_tools.jl"))

infile, outfile = ARGS[1], ARGS[2]
nroots = parse(Int, get(ENV, "TPSCI_NROOTS", "8"))
M = parse(Int, get(ENV, "TPSCI_M", "100"))
delta = parse(Int, get(ENV, "TPSCI_DELTA", "2"))
thresh_cipsi = parse(Float64, get(ENV, "TPSCI_THRESH_CIPSI", "1e-4"))
thresh_foi = parse(Float64, get(ENV, "TPSCI_THRESH_FOI", "1e-6"))
spin_tol = parse(Float64, get(ENV, "TPSCI_SPIN_TOL", "0.1"))

d = npzread(infile)
ints = InCoreInts(first(d["h0"]), d["h1"], d["h2"])
nclus = maximum(d["cluster_index"])
clusters = [MOCluster(i, findall(==(i), d["cluster_index"])) for i in 1:nclus]
init_fspace = FockConfig([Tuple(d["fspace"][i, :]) for i in 1:nclus])
d1 = RDM1(d["rdm1a"], d["rdm1b"])
n = n_orb(ints)
nel = sum(sum(f) for f in init_fspace.config)
@printf(" %d clusters %s, (%de, %do), %d roots, M = %d, thresh %.1e\n", nclus, [length(c) for c in clusters], nel, n,
        nroots, M, thresh_cipsi)

cluster_bases = TPSChem.compute_cluster_eigenbasis_spin(ints, clusters, d1, [delta for i in 1:nclus], init_fspace,
                                                        max_roots=M, verbose=0)
clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints)
TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

# start: the reference and its spin-flip partners, plus single cluster excitations, so that the excited states of
# interest (local excitations, charge transfer, triplet pairs) are reachable from the first iteration
ci = TPSChem.TPSCIstate(clusters, init_fspace, R=nroots)
ci = TPSChem.add_spin_focksectors(ci)
for i in 1:nclus, k in 2:min(nroots, size(cluster_bases[i][init_fspace[i]].vectors, 2))
    cfg = ones(Int, nclus); cfg[i] = k
    ci[init_fspace][TPSChem.ClusterConfig(cfg)] = zeros(nroots)
end
TPSChem.eye!(ci)
e0, v0 = TPSChem.tpsci_ci(ci, cluster_ops, clustered_ham, thresh_cipsi=thresh_cipsi, thresh_foi=thresh_foi,
                          thresh_asci=-1, conv_thresh=parse(Float64, get(ENV, "TPSCI_CONV", "1e-7")),
                          ci_conv=parse(Float64, get(ENV, "TPSCI_CI_CONV", "1e-9")), nbody=4)
s2 = TPSChem.compute_expectation_value_parallel(v0, cluster_ops, TPSChem.extract_S2(clusters))
s2 = [x for x in (s2 isa AbstractMatrix ? diag(s2) : s2)]
for r in 1:nroots
    @printf(" root %2d  E = %.10f  <S^2> = %.4f\n", r, e0[r] + ints.h0, s2[r])
end

pairs = [(I, J) for I in 1:nroots for J in I:nroots if abs(s2[I] - s2[J]) < spin_tol]
offsets = cumsum([0; [length(c) for c in clusters[1:end-1]]])
@time G = intercluster_2rdm(v0, cluster_ops, offsets, progress=false, pairs=pairs)
@time oncluster_2rdm!(G, v0, cluster_bases, offsets, progress=false, pairs=pairs)
D = zeros(n, n, length(pairs))
for (ch, (I, J)) in enumerate(pairs)
    I == J || symmetrize_transition!(view(G, :, :, :, :, ch))
    D[:, :, ch] .= sum(G[:, q, :, q, ch] for q in 1:n) ./ (nel - 1)
    w = ints.h0 * (I == J) + sum(ints.h1 .* D[:, :, ch]) + 0.5 * sum(ints.h2 .* permutedims(G[:, :, :, :, ch], (1, 3, 2, 4)))
    ref = I == J ? e0[I] + ints.h0 : 0.0
    @printf(" <%d|H|%d> from RDMs %.10f (TPSCI %.10f, diff %.1e)\n", I, J, w, ref, w - ref)
end
npzwrite(outfile, Dict("energies" => collect(e0) .+ ints.h0, "s2" => s2, "pairs" => hcat([[p[1], p[2]] for p in pairs]...)',
                       "dm1" => D, "dm2" => G))
jldsave(splitext(outfile)[1] * ".jld2"; v0, e0, cluster_bases, s2)
@printf(" wrote %s (%d channels)\n", outfile, length(pairs))
