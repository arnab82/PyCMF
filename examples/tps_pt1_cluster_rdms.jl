# First-order (PT1) correlation between clusters in the cluster-state basis, for active-space selection with cMF
# (pycmf.active_space): the cluster-block RDMs of psi0 + psi1, with psi0 the lowest state of the spin-complete
# reference space (cluster ground states in the reference Fock sector and its spin-flip partners) and psi1 its
# first-order correction (cMF H0, first-order interacting space of TPSChem, reference space projected out), without
# determinants. TPSPT1_SHIFT: level shift of the denominators (default 0).
#
#   julia --project=/path/to/TPSChem.jl -t auto tps_pt1_cluster_rdms.jl cmf_export.npz pt1_blocks.npz
#
# cmf_export.npz is written by CMF.export_fermicg. Settings (environment variables): TPSPT1_M cluster states per Fock
# sector (default 30), TPSPT1_DELTA charge range of the cluster bases (2: a two-electron operator moves up to two
# electrons into one cluster), TPSPT1_THRESH_FOI (1e-5), TPSPT1_NBODY (4), TPSPT1_NROOTS (1): states of the reference
# space treated (e.g. 4 for the S = 0..3 ladder of two S = 3/2; root r > 1 is written with the suffix _r<r>).
# Output: for each cluster i (1-based) d1a_i, d1b_i (n_i, n_i) and d2ab_i (n_i^4) of the normalized psi0 + psi1, in
# PySCF's conventions (d2ab[p,q,r,s] = <p_a' r_b' s_b q_a>), the same of psi0 alone (d1a0_i, ...), and e2, nfois.
using TPSChem
using TPSChem.QCBase
using TPSChem.InCoreIntegrals
using TPSChem.RDM
using NPZ
using Printf
using LinearAlgebra
using StaticArrays
include(joinpath(@__DIR__, "tpsci_rdm_tools.jl"))

infile, outfile = ARGS[1], ARGS[2]
M = parse(Int, get(ENV, "TPSPT1_M", "30"))
delta = parse(Int, get(ENV, "TPSPT1_DELTA", "2"))
thresh_foi = parse(Float64, get(ENV, "TPSPT1_THRESH_FOI", "1e-5"))
nbody = parse(Int, get(ENV, "TPSPT1_NBODY", "4"))

d = npzread(infile)
ints = InCoreInts(first(d["h0"]), d["h1"], d["h2"])
nclus = maximum(d["cluster_index"])
clusters = [MOCluster(i, findall(==(i), d["cluster_index"])) for i in 1:nclus]
init_fspace = FockConfig([Tuple(d["fspace"][i, :]) for i in 1:nclus])
d1 = RDM1(d["rdm1a"], d["rdm1b"])
@printf(" %d clusters %s, %d orbitals, reference %s, M = %d, delta = %d, thresh_foi = %.1e\n", nclus,
        [length(c) for c in clusters], n_orb(ints), init_fspace, M, delta, thresh_foi)
flush(stdout)

@time cluster_bases = TPSChem.compute_cluster_eigenbasis_spin(ints, clusters, d1, [delta for i in 1:nclus], init_fspace,
                                                              max_roots=M, verbose=0)
clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
@time cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints)
TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)
flush(stdout)

# psi0: lowest state of the spin-complete reference space (cluster ground states in the reference Fock sector and its
# spin-flip partners, e.g. (2,1)(1,2) for (3,0)(0,3)): the spin-flip configurations have the same H0 energy as the
# reference, so they must be in the reference space, not in the first-order space (intruders)
nroots = parse(Int, get(ENV, "TPSPT1_NROOTS", "1"))
ref = TPSChem.TPSCIstate(clusters, init_fspace, R=nroots)
ref = TPSChem.add_spin_focksectors(ref)
TPSChem.eye!(ref)
e_ref, psi0 = TPSChem.tps_ci_direct(ref, cluster_ops, clustered_ham)
s2 = TPSChem.compute_expectation_value_parallel(psi0, cluster_ops, TPSChem.extract_S2(clusters))
for r in 1:nroots
    @printf(" reference space: %d configurations, root %d E = %.10f  <S^2> = %.4f\n", length(psi0), r, e_ref[r],
            s2[r] isa AbstractMatrix ? s2[r, r] : s2[r])
end

# first-order wavefunction with cMF H0: psi1 = sum_k |k> <k|H|psi0> / (E0 - E0_k), k outside the reference space
@time sig = TPSChem.open_matvec_thread(psi0, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
TPSChem.project_out!(sig, ref)
clustered_ham_0 = TPSChem.extract_1body_operator(clustered_ham, op_string="Hcmf")
Hd = TPSChem.compute_diagonal(sig, cluster_ops, clustered_ham_0)
E0s = TPSChem.compute_expectation_value_parallel(psi0, cluster_ops, clustered_ham_0)
shift = parse(Float64, get(ENV, "TPSPT1_SHIFT", "0.0"))
SV = TPSChem.get_vector(sig)
V1 = zeros(size(SV))
e2 = zeros(nroots)
for r in 1:nroots
    E0 = E0s[r] isa Number ? E0s[r] : E0s[r, r]
    den = E0 .- Hd .- shift
    small = abs.(den) .< 1e-6
    V1[:, r] .= ifelse.(small, 0.0, SV[:, r] ./ ifelse.(small, 1.0, den))
    e2[r] = sum(SV[:, r] .* V1[:, r])
    any(small) && @printf(" root %d: %d first-order configurations within 1e-6 Eh of E0 skipped\n", r, count(small))
end
psi1 = deepcopy(sig)
TPSChem.set_vector!(psi1, V1)
nfois = length(psi1)
psi = deepcopy(psi1)
TPSChem.add!(psi, psi0)
nrm = TPSChem.norm(psi)
for r in 1:nroots
    TPSChem.scale!(psi, 1 / nrm[r], root=r)
    @printf(" PT1 root %d: %d configurations in the first-order interacting space, E2 = %.10f, |psi0 + psi1| = %.6f\n",
            r, nfois, e2[r], nrm[r])
end
flush(stdout)

out = Dict{String,Any}("e2" => e2, "nfois" => [nfois], "norm" => collect(nrm[1:nroots]), "e_ref" => collect(e_ref[1:nroots]))
for r in 1:nroots
    tag = r == 1 ? "" : "_r$r"               # root 1 without suffix (load_cluster_rdms default)
    @time blocks = cluster_block_rdms(psi, cluster_bases, r)
    blocks0 = cluster_block_rdms(psi0, cluster_bases, r)
    for i in 1:nclus
        a, b, ab = blocks[i]
        a0, b0, ab0 = blocks0[i]
        out["d1a_$i$tag"], out["d1b_$i$tag"], out["d2ab_$i$tag"] = a, b, ab
        out["d1a0_$i$tag"], out["d1b0_$i$tag"], out["d2ab0_$i$tag"] = a0, b0, ab0
        occ = reverse(eigvals(Symmetric(a + b)))
        @printf(" root %d cluster %2d %s: N = %.6f, natural occupations %s\n", r, i, init_fspace[i], tr(a + b),
                join([@sprintf("%.4f", x) for x in occ], " "))
    end
end
npzwrite(outfile, out)
@printf(" wrote %s\n", outfile)
