# Checks of tpsci_rdm_tools.jl against TPSChem on a small TPSCI state (H12, 5 clusters, 3 roots):
#   - cluster_2rdm_gram against FCI.compute_rdm1_rdm2 (mixed states, empty and full sectors)
#   - intercluster_2rdm (factorized and pair-by-pair kernels) against TPSChem._accumulate_2rdm_intercluster_full!
#   - oncluster_2rdm! against TPSChem._accumulate_2rdm_oncluster! (Ppqsr tables)
#   - the energies of the RDMs against the TPSCI energies (TPSChem's leave out h0)
#
#   julia --project=/path/to/TPSChem.jl -t 4 test_tpsci_rdm_tools.jl /path/to/TPSChem.jl/test/_testdata_cmf_h12_64bit.jld2
using TPSChem, TPSChem.QCBase, JLD2, LinearAlgebra, Printf, Test
include(joinpath(@__DIR__, "tpsci_rdm_tools.jl"))
const FCI = TPSChem.ActiveSpaceSolvers.FCI

function fci_reference(ans, Φ, w)
    no = ans.no
    G = zeros(no, no, no, no)
    for k in eachindex(w)
        _, _, aa, bb, ab = FCI.compute_rdm1_rdm2(ans, Φ[:, k], Φ[:, k])
        for s in 1:no, r in 1:no, q in 1:no, p in 1:no
            G[p, q, r, s] += w[k] * (aa[p, r, q, s] + bb[p, r, q, s] + ab[p, r, q, s] + ab[q, s, p, r])
        end
    end
    return G
end

@testset "cluster_2rdm_gram" begin
    for (no, na, nb) in [(6, 3, 3), (6, 4, 1), (6, 1, 0), (6, 0, 2), (5, 2, 3), (6, 6, 5), (4, 1, 1)]
        ans = FCI.FCIAnsatz(no, na, nb)
        Φ, w = randn(ans.dim, 3), [0.5, 0.3, -1e-3]
        @test maximum(abs.(cluster_2rdm_gram(no, na, nb, Φ, w) - fci_reference(ans, Φ, w))) < 1e-12
    end
end

@testset "TPSCI RDMs, H12" begin
    @load ARGS[1] ints clusters init_fspace d1 cluster_bases
    clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
    cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints)
    TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)
    R = 3
    ref = TPSChem.FockConfig(init_fspace)
    v = TPSChem.TPSCIstate(clusters, ref, R=R)
    for k in 1:R
        v[ref][TPSChem.ClusterConfig([1, 1, k, 1, 1])] = zeros(R)
    end
    TPSChem.eye!(v)
    e0, v0 = TPSChem.tpsci_ci(v, cluster_ops, clustered_ham, thresh_cipsi=1e-3, thresh_foi=1e-6, thresh_asci=-1,
                              conv_thresh=1e-6)
    n = n_orb(ints)
    offsets = cumsum([0; [length(c) for c in clusters[1:end-1]]])

    Gref = zeros(n, n, n, n, R, R)
    TPSChem._accumulate_2rdm_intercluster_full!(Gref, v0, v0, cluster_ops, offsets)
    G = intercluster_2rdm(v0, cluster_ops, offsets, progress=false)
    Gslow = intercluster_2rdm(v0, cluster_ops, offsets, progress=false, kernel=_combination_2rdm!)
    @test maximum(maximum(abs.(G[:, :, :, :, r] - Gref[:, :, :, :, r, r])) for r in 1:R) < 1e-12
    @test maximum(abs.(G - Gslow)) < 1e-12

    TPSChem.add_spinfree_2rdm_ops!(cluster_ops, cluster_bases)
    Gon_ref = zeros(n, n, n, n, R, R)
    TPSChem._accumulate_2rdm_oncluster!(Gon_ref, v0, v0, cluster_ops, offsets)
    Gon = zeros(n, n, n, n, R)
    oncluster_2rdm!(Gon, v0, cluster_bases, offsets, progress=false)
    @test maximum(maximum(abs.(Gon[:, :, :, :, r] - Gon_ref[:, :, :, :, r, r])) for r in 1:R) < 1e-12

    γa, γb = TPSChem.compute_1rdm(v0, cluster_ops)
    nel = sum(sum(f) for f in init_fspace)
    for r in 1:R
        dm1 = γa[:, :, r, r] + γb[:, :, r, r]
        dm2 = G[:, :, :, :, r] + Gon[:, :, :, :, r]
        # the 1RDM from the 2RDM, as with TPSCI_1RDM=trace
        @test maximum(abs.(sum(dm2[:, q, :, q] for q in 1:n) / (nel - 1) - dm1)) < 1e-12
        e = sum(ints.h1 .* dm1) + 0.5 * sum(ints.h2 .* permutedims(dm2, (1, 3, 2, 4)))
        @printf(" root %d  E(TPSCI) %.12f  E(RDMs) %.12f\n", r, e0[r], e)
        @test abs(e - e0[r]) < 1e-10
    end
end
