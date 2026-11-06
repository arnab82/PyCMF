# First-order (PT1) correlation between clusters in the cluster-state basis, for active-space selection with cMF
# (pycmf.active_space): the cluster-block RDMs of psi0 + psi1, with psi0 the cMF product state and psi1 its first-order
# correction (TPSChem.compute_pt1_wavefunction, cMF H0, first-order interacting space), without determinants.
#
#   julia --project=/path/to/TPSChem.jl -t auto tps_pt1_cluster_rdms.jl cmf_export.npz pt1_blocks.npz
#
# cmf_export.npz is written by CMF.export_fermicg. Settings (environment variables): TPSPT1_M cluster states per Fock
# sector (default 30), TPSPT1_DELTA charge range of the cluster bases (2: a two-electron operator moves up to two
# electrons into one cluster), TPSPT1_THRESH_FOI (1e-5), TPSPT1_NBODY (4).
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

ref = TPSChem.TPSCIstate(clusters, init_fspace, R=1)
TPSChem.eye!(ref)
@time e2, psi1 = TPSChem.compute_pt1_wavefunction(ref, cluster_ops, clustered_ham, H0="Hcmf", nbody=nbody,
                                                  thresh_foi=thresh_foi)
nfois = length(psi1)
psi = deepcopy(psi1)
TPSChem.add!(psi, ref)
nrm = TPSChem.norm(psi)
TPSChem.scale!(psi, 1 / nrm[1])
@printf(" PT1: %d configurations in the first-order interacting space, E2 = %.10f, |psi0 + psi1| = %.6f\n",
        nfois, e2[1], nrm[1])
flush(stdout)

out = Dict{String,Any}("e2" => collect(e2), "nfois" => [nfois], "norm" => [nrm[1]])
@time blocks = cluster_block_rdms(psi, cluster_bases)
blocks0 = cluster_block_rdms(ref, cluster_bases)
for i in 1:nclus
    a, b, ab = blocks[i]
    a0, b0, ab0 = blocks0[i]
    out["d1a_$i"], out["d1b_$i"], out["d2ab_$i"] = a, b, ab
    out["d1a0_$i"], out["d1b0_$i"], out["d2ab0_$i"] = a0, b0, ab0
    occ = reverse(eigvals(Symmetric(a + b)))
    @printf(" cluster %2d %s: N = %.6f, natural occupations %s\n", i, init_fspace[i], tr(a + b),
            join([@sprintf("%.4f", x) for x in occ], " "))
end
npzwrite(outfile, out)
@printf(" wrote %s\n", outfile)
