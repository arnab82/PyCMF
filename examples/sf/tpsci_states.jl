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
# <states>.jld2. With one cluster per chromophore: class_weights (R, R, K) = sum over the configurations of each
# diabatic class (GS, LE_i, CT_i_j, TT_i_j, other; names in <states>_classes.txt) of c_I c_J.
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
# ---------------------------------------------------------------- diabatic character (one cluster per chromophore)
# label the cluster states: G ground, S1 lowest singlet-like excitation of the neutral sector (not degenerate with a
# state of the (n+1, n-1) sector, i.e. not an Ms = 0 component of a triplet or quintet), T lowest triplet (all Ms
# components), A / Cn anion / cation ground (both Ms); configurations -> GS, LE_i, CT_i_j (i anion, j cation),
# TT_i_j, T_i (local triplet), other; W[k][I, J] = sum over the configurations of class k of c_I c_J
function cluster_labels(cb, ref; tol=1e-6)
    na, nb = Int(ref[1]), Int(ref[2])
    sec(a, b) = (Int16(a), Int16(b))
    en(a, b) = haskey(cb.basis, sec(a, b)) ? cb.basis[sec(a, b)].energies : Float64[]
    lab = Dict{Tuple{Tuple{Int16,Int16},Int},Symbol}()
    lab[(sec(na, nb), 1)] = :G
    hs = en(na + 1, nb - 1)
    if !isempty(hs)
        for (a, b) in ((na + 1, nb - 1), (na, nb), (na - 1, nb + 1)), (k, e) in enumerate(en(a, b))
            abs(e - hs[1]) < tol && (lab[(sec(a, b), k)] = :T)
        end
    end
    for (k, e) in enumerate(en(na, nb))
        k == 1 && continue
        any(abs(e - h) < tol for h in hs) && continue
        lab[(sec(na, nb), k)] = :S1
        break
    end
    for (a, b) in ((na + 1, nb), (na, nb + 1)); isempty(en(a, b)) || (lab[(sec(a, b), 1)] = :A); end
    for (a, b) in ((na - 1, nb), (na, nb - 1)); isempty(en(a, b)) || (lab[(sec(a, b), 1)] = :Cn); end
    return lab
end
labels = [cluster_labels(cluster_bases[i], init_fspace[i]) for i in 1:nclus]
function config_class(fock, config)
    l = [get(labels[i], ((Int16(fock[i][1]), Int16(fock[i][2])), Int(config[i])), :X) for i in 1:nclus]
    rest(idx...) = all(l[k] == :G for k in 1:nclus if !(k in idx))
    all(==(:G), l) && return "GS"
    s1, t, a, c = findall(==(:S1), l), findall(==(:T), l), findall(==(:A), l), findall(==(:Cn), l)
    length(s1) == 1 && rest(s1[1]) && return "LE_$(s1[1])"
    length(a) == 1 && length(c) == 1 && rest(a[1], c[1]) && return "CT_$(a[1])_$(c[1])"
    length(t) == 2 && rest(t...) && return "TT_$(t[1])_$(t[2])"
    length(t) == 1 && rest(t[1]) && return "T_$(t[1])"
    return "other"
end
W = Dict{String,Matrix{Float64}}()
for (fock, configs) in v0.data, (config, coeff) in configs
    k = config_class(fock, config)
    w = get!(() -> zeros(nroots, nroots), W, k)
    w .+= coeff * coeff'
end
cnames = sort(collect(keys(W)))
for r in 1:nroots
    @printf(" root %2d character: %s\n", r, join(["$k $(round(W[k][r, r], digits=3))" for k in cnames if W[k][r, r] > 0.005], ", "))
end
open(splitext(outfile)[1] * "_classes.txt", "w") do f
    foreach(k -> println(f, k), cnames)
end
npzwrite(outfile, Dict("energies" => collect(e0) .+ ints.h0, "s2" => s2, "pairs" => hcat([[p[1], p[2]] for p in pairs]...)',
                       "dm1" => D, "dm2" => G, "class_weights" => cat([W[k] for k in cnames]..., dims=3)))
jldsave(splitext(outfile)[1] * ".jld2"; v0, e0, cluster_bases, s2)
@printf(" wrote %s (%d channels)\n", outfile, length(pairs))
