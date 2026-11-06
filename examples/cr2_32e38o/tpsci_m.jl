# TPSCI (TPSChem.jl) of the (32e, 38o) Cr2 problem from make_integrals.py at one M, with the threshold schedule of the
# 2024 Faraday Discussions run (cr2/m100/tpsci_05.jl), saving the wavefunctions with their cluster bases and the RDMs
# of every root for PDFT (tpsci_pdft.py).
#
#   julia --project=/path/to/TPSChem.jl -t auto tpsci_m.jl svp/cmf_32e38o.npz svp/M100
#
# Settings (environment variables):
#   TPSCI_M          cluster states per Fock sector (100)
#   TPSCI_DELTA      charge range of the cluster bases (3, as the Faraday run)
#   TPSCI_NROOTS     roots (4: the S = 0..3 ladder of two S = 3/2)
#   TPSCI_THRESHS    thresh_cipsi schedule, comma separated (1e-3,8e-4,6e-4,4e-4,2e-4); thresh_spin = thresh_cipsi,
#                    thresh_foi = thresh_cipsi / 10, as the Faraday run
#   TPSCI_PT2        1: PT2 correction after each threshold (compute_pt2_energy), 0: skip (1)
#   TPSCI_RDMS       all | last | none: thresholds whose RDMs are computed (all)
#   TPSCI_MAX_MEM_CI GB for the dense CI Hamiltonian in tpsci_ci (20). Beyond it tpsci_ci switches to TPSChem's
#                    matrix-free Davidson, which is very slow for ~1e5 configurations: the Faraday run used 700
#                    (97k configurations at 2e-4 need a 76 GB dense H)
#
# Output in the directory (second argument):
#   cluster_bases.jld2    cluster_bases, clusters, init_fspace, ints, M, delta: the basis the states are expanded in.
#                         Keep it with the states: cluster states rebuilt elsewhere (other BLAS) can differ in sign,
#                         which scrambles the saved coefficients
#   tpsci_tNN.jld2        v0a (TPSCIstate, all roots), e0a, e2a, thresh, M, after threshold NN
#   rdms_tNN.npz          energies (R,) with h0, e2a, dm1 (n, n, R) spin-summed, dm2 (n, n, n, n, R) spin-free,
#                         G[p,q,r,s] = <p'q'sr>, in the orbital order of cmf_32e38o.npz; root-diagonal only
#                         (tpsci_rdm_tools.jl: inter-cluster blocks threaded and factorized, on-cluster blocks from
#                         natural cluster states; 1RDM from the 2RDM, sum_q G[p,q,r,q] = (N-1) D[p,r])
#   summary.txt           energies, PT2, J of each threshold
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

infile, outdir = ARGS[1], ARGS[2]
mkpath(outdir)
M = parse(Int, get(ENV, "TPSCI_M", "100"))
delta = parse(Int, get(ENV, "TPSCI_DELTA", "3"))
nroots = parse(Int, get(ENV, "TPSCI_NROOTS", "4"))
threshs = parse.(Float64, split(get(ENV, "TPSCI_THRESHS", "1e-3,8e-4,6e-4,4e-4,2e-4"), ","))
do_pt2 = get(ENV, "TPSCI_PT2", "1") == "1"
rdm_mode = get(ENV, "TPSCI_RDMS", "all")
max_mem_ci = parse(Float64, get(ENV, "TPSCI_MAX_MEM_CI", "20"))
const CM = 219474.6313705

d = npzread(infile)
ints = InCoreInts(first(d["h0"]), d["h1"], d["h2"])
nclus = maximum(d["cluster_index"])
clusters = [MOCluster(i, findall(==(i), d["cluster_index"])) for i in 1:nclus]
init_fspace = FockConfig([Tuple(d["fspace"][i, :]) for i in 1:nclus])
d1 = RDM1(d["rdm1a"], d["rdm1b"])
n = n_orb(ints)
nel = sum(sum(f) for f in init_fspace.config)
@printf(" %d clusters %s, (%de, %do), reference %s\n M = %d, delta = %d, roots = %d, thresholds %s, max_mem_ci %.0f GB\n",
        nclus, [length(c) for c in clusters], nel, n, init_fspace, M, delta, nroots, threshs, max_mem_ci)
flush(stdout)

@time cluster_bases = TPSChem.compute_cluster_eigenbasis_spin(ints, clusters, d1, [delta for i in 1:nclus], init_fspace,
                                                              max_roots=M, verbose=1)
clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
@time cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints)
TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)
jldsave(joinpath(outdir, "cluster_bases.jld2"); cluster_bases, clusters, init_fspace, ints, M, delta)
offsets = cumsum([0; [length(c) for c in clusters[1:end-1]]])

"""J of H = -2J S_A.S_B from E(S) = E0 - J S(S+1), S from the roots' <S^2> (lowest root of each S)"""
function fit_j(e, s2)
    S = round.(Int, (sqrt.(1 .+ 4 .* max.(s2, 0)) .- 1) ./ 2)
    levels = Dict{Int,Float64}()
    for (ei, si) in zip(e, S)
        levels[si] = min(get(levels, si, Inf), ei)
    end
    ss = sort(collect(keys(levels)))
    length(ss) < 2 && return NaN
    A = hcat(ones(length(ss)), -[s * (s + 1) for s in ss])
    return (A \ [levels[s] for s in ss])[2] * CM
end

ci_vector = TPSChem.TPSCIstate(clusters, init_fspace, R=nroots)
ci_vector = TPSChem.add_spin_focksectors(ci_vector)
TPSChem.eye!(ci_vector)
eci, v0a = TPSChem.tps_ci_direct(ci_vector, cluster_ops, clustered_ham)
S2op = TPSChem.extract_S2(clusters)
summary = open(joinpath(outdir, "summary.txt"), "a")
@printf(summary, "# M = %d, delta = %d; energies without h0 = %.10f; J (cm-1) for H = -2J SA.SB\n", M, delta, ints.h0)

for (k, thr) in enumerate(threshs)
    tag = @sprintf("t%02d", k - 1)
    e0a, v0a = TPSChem.tpsci_ci(v0a, cluster_ops, clustered_ham, thresh_cipsi=thr, thresh_spin=thr, thresh_foi=thr / 10,
                                ci_max_iter=150, ci_lindep_thresh=1e-12, conv_thresh=1e-6, max_iter=20, nbody=4,
                                max_mem_ci=max_mem_ci)
    s2 = TPSChem.compute_expectation_value_parallel(v0a, cluster_ops, S2op)
    s2 = [x isa Number ? x : 0.0 for x in (s2 isa AbstractMatrix ? diag(s2) : s2)]
    e2a = do_pt2 ? TPSChem.compute_pt2_energy(v0a, cluster_ops, clustered_ham) : zeros(nroots)
    jldsave(joinpath(outdir, "tpsci_$tag.jld2"); v0a, e0a, e2a, s2, thresh=thr, M, delta)
    j_var, j_pt2 = fit_j(collect(e0a), s2), fit_j(collect(e0a) .+ collect(e2a), s2)
    @printf(" threshold %.1e: %d configurations; E(var) %s; <S^2> %s; J(var) = %.2f, J(var+PT2) = %.2f cm-1\n",
            thr, length(v0a), round.(collect(e0a) .+ ints.h0, digits=8), round.(s2, digits=3), j_var, j_pt2)
    @printf(summary, "%s thresh %.1e ncfg %d E %s E2 %s S2 %s Jvar %.3f Jpt2 %.3f\n", tag, thr, length(v0a),
            join(@sprintf("%.10f", x + ints.h0) for x in e0a), join(@sprintf("%.10f", x) for x in e2a),
            join(@sprintf("%.4f", x) for x in s2), j_var, j_pt2)
    flush(summary); flush(stdout)

    (rdm_mode == "all" || (rdm_mode == "last" && k == length(threshs))) || continue
    @time dm2 = intercluster_2rdm(v0a, cluster_ops, offsets)
    @time oncluster_2rdm!(dm2, v0a, cluster_bases, offsets)
    dm1 = cat([sum(dm2[:, q, :, q, r] for q in 1:n) / (nel - 1) for r in 1:nroots]..., dims=3)
    e_rdm = [ints.h0 + sum(ints.h1 .* dm1[:, :, r]) + 0.5 * sum(ints.h2 .* permutedims(dm2[:, :, :, :, r], (1, 3, 2, 4)))
             for r in 1:nroots]
    for r in 1:nroots
        @printf("   root %d  E(TPSCI) = %.10f  E(RDMs) = %.10f  diff %.1e  Tr(dm1) = %.6f\n", r, e0a[r] + ints.h0,
                e_rdm[r], e_rdm[r] - e0a[r] - ints.h0, tr(dm1[:, :, r]))
    end
    maximum(abs.(e_rdm .- e0a .- ints.h0)) < 1e-6 || @warn "RDM energies differ from the TPSCI energies"
    npzwrite(joinpath(outdir, "rdms_$tag.npz"), Dict("energies" => collect(e0a) .+ ints.h0, "e2a" => collect(e2a),
                                                     "s2" => s2, "dm1" => dm1, "dm2" => dm2, "thresh" => [thr],
                                                     "M" => [M], "ncfg" => [length(v0a)]))
    @printf(" wrote %s\n", joinpath(outdir, "rdms_$tag.npz"))
    flush(stdout)
end
close(summary)
