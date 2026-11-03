# Root-diagonal RDMs of a TPSCI state with TPSChem.jl, used by tpsci_rdms_from_jld2.jl.
#
# intercluster_2rdm(v, cluster_ops, offsets): the spin-free 2RDM blocks whose four indices are not all on one cluster,
# G[p,q,r,s,root] = <p'q'sr>, for each root (no transition RDMs between roots). It is the loop of TPSChem's
# _accumulate_2rdm_intercluster_full!, with its local operators, signs and canonical operator strings, but
#   - the operator product of a pair of configurations is built once and added to all roots with their weights
#     c_r(bra) c_r(ket), instead of once per pair of roots (R^2 times),
#   - the cluster combinations (p,q,s,r on clusters I,J,K,L) are distributed over the Julia threads, each with its
#     own accumulator,
#   - a progress line is printed after each cluster combination (count, elapsed time, estimated time left).
# oncluster_2rdm!(G, v, cluster_bases, offsets): the blocks with all four indices on one cluster, from the cluster
# density matrices diagonalized into natural cluster states (an FCI 2RDM per natural state).
using TPSChem
using TPSChem.QCBase
using LinearAlgebra
using StaticArrays
using Printf

# leaf: G[root, p, q, r, s] += coeff * w[root], with (p, q, s, r) the orbitals of the roles 1..4 of p'q'sr
function _op_product_rec!(G::Array{T,5}, local_ops, role_lists, norbs, offs, s_active, t_active, role_orbs,
                          coeff::T, w::AbstractVector{T}, k::Int) where T
    if k > length(local_ops)
        p, q, s, r = role_orbs[1], role_orbs[2], role_orbs[3], role_orbs[4]
        @inbounds @simd for root in eachindex(w)
            G[root, p, q, r, s] += coeff * w[root]
        end
        return
    end
    opmat, roles, norb_i, off_i = local_ops[k], role_lists[k], norbs[k], offs[k]
    sk, tk = s_active[k], t_active[k]
    @inbounds for flat in 1:size(opmat, 1)
        opval = opmat[flat, sk, tk]
        iszero(opval) && continue
        x = flat - 1
        for i in eachindex(roles)
            role_orbs[roles[i]] = off_i + (x % norb_i) + 1
            x = div(x, norb_i)
        end
        _op_product_rec!(G, local_ops, role_lists, norbs, offs, s_active, t_active, role_orbs, coeff * opval, w, k + 1)
    end
    return
end

# the contributions of one cluster combination (cp, cq, cs, cr), all spin cases, to G[root, p, q, r, s]
function _combination_2rdm!(G::Array{T,5}, v::TPSChem.TPSCIstate{T,N,R}, cluster_ops, offsets,
                            term_clusters) where {T,N,R}
    roles0 = [1, 2, 3, 4]  # p, q, s, r
    perm, countswap = TPSChem.bubble_sort(term_clusters)
    sorted_clusters = term_clusters[perm]
    perm_sign = isodd(countswap) ? -1 : 1
    w = zeros(T, R)
    role_orbs = zeros(Int, 4)
    for (spin_ops0, fock_deltas0) in TPSChem._TPSCI_2RDM_SPIN_CASES
        spin_ops = collect(spin_ops0)[perm]
        fock_deltas = collect(fock_deltas0)[perm]
        sorted_roles = roles0[perm]
        active_clusters, active_indices, opstrings = MOCluster[], Int[], String[]
        cluster_deltas, role_lists = Tuple{Int,Int}[], Vector{Int}[]
        canonical_sign = 1
        pos = 1
        while pos <= 4
            ci = sorted_clusters[pos]
            opstring, da, db, roles = "", 0, 0, Int[]
            while pos <= 4 && sorted_clusters[pos].idx == ci.idx
                opstring *= spin_ops[pos]
                da += fock_deltas[pos][1]
                db += fock_deltas[pos][2]
                push!(roles, sorted_roles[pos])
                pos += 1
            end
            opstring, roles, local_sign = TPSChem._canonicalize_2rdm_opstring(opstring, roles)
            canonical_sign *= local_sign
            push!(active_clusters, ci); push!(active_indices, ci.idx); push!(opstrings, opstring)
            push!(cluster_deltas, (da, db)); push!(role_lists, roles)
        end
        inactive_indices = [idx for idx in 1:N if !(idx in active_indices)]
        parities = [isodd(length(o)) ? 1 : 0 for o in opstrings]
        norbs = [length(ci) for ci in active_clusters]
        offs = [offsets[ci.idx] for ci in active_clusters]

        for (fock_ket, configs_ket) in v.data
            new_focks = Vector{Tuple{Int,Int}}(undef, length(active_indices))
            valid = true
            for i in eachindex(active_indices)
                ci = active_clusters[i]
                fo = fock_ket[ci.idx]
                fn = (fo[1] + cluster_deltas[i][1], fo[2] + cluster_deltas[i][2])
                if !TPSChem._valid_2rdm_fock(fn, ci)
                    valid = false
                    break
                end
                new_focks[i] = fn
            end
            valid || continue
            fock_bra = TPSChem.replace(fock_ket, active_indices, new_focks)
            haskey(v.data, fock_bra) || continue
            configs_bra = v.data[fock_bra]
            local_ops = TPSChem._lookup_2rdm_local_ops(cluster_ops, active_clusters, opstrings, fock_bra, fock_ket)
            local_ops === nothing && continue

            groups(configs) = begin
                g = Dict{Vector{Int16},Vector{Tuple{Vector{Int16},MVector{R,T}}}}()
                for (config, coeff) in configs
                    push!(get!(g, [config[i] for i in inactive_indices], Tuple{Vector{Int16},MVector{R,T}}[]),
                          ([config[i] for i in active_indices], coeff))
                end
                g
            end
            bra_groups, ket_groups = groups(configs_bra), groups(configs_ket)
            term_sign = convert(T, perm_sign * canonical_sign *
                                   TPSChem._2rdm_state_sign(active_clusters, parities, fock_ket))
            for (key, bra_list) in bra_groups
                haskey(ket_groups, key) || continue
                ket_list = ket_groups[key]
                for (s_active, c_bra) in bra_list, (t_active, c_ket) in ket_list
                    nz = false
                    for r in 1:R
                        w[r] = c_bra[r] * c_ket[r]
                        nz |= !iszero(w[r])
                    end
                    nz || continue
                    _op_product_rec!(G, local_ops, role_lists, norbs, offs, s_active, t_active, role_orbs,
                                     term_sign, w, 1)
                end
            end
        end
    end
    return G
end

"""
    intercluster_2rdm(v, cluster_ops, offsets; progress=true) -> G (n, n, n, n, R)

Inter-cluster blocks of the spin-free 2RDM of each root, G[p,q,r,s,root] = <p'q'sr>, threaded over cluster
combinations, with a progress line per combination
"""
function intercluster_2rdm(v::TPSChem.TPSCIstate{T,N,R}, cluster_ops, offsets; progress=true) where {T,N,R}
    clusters = v.clusters
    n = sum(length(c) for c in clusters)
    combos = [[cp, cq, cs, cr] for cp in clusters, cq in clusters, cs in clusters, cr in clusters
              if length(unique([cp.idx, cq.idx, cs.idx, cr.idx])) > 1]
    # most expensive first (fewer distinct clusters = larger local operators), for a better load balance
    sort!(combos, by=c -> length(unique([x.idx for x in c])))
    queue = Channel{Int}(length(combos))
    foreach(i -> put!(queue, i), eachindex(combos))
    close(queue)
    lk, done, t0 = ReentrantLock(), Ref(0), time()
    nw = Threads.nthreads()
    progress && @printf(" inter-cluster 2RDM: %d cluster combinations, %d roots, %d threads\n", length(combos), R, nw)
    tasks = map(1:nw) do _
        Threads.@spawn begin
            Gt = zeros(T, R, n, n, n, n)   # (not G: that would capture the G of the enclosing function)
            for i in queue
                ti = time()
                _combination_2rdm!(Gt, v, cluster_ops, offsets, combos[i])
                if progress
                    lock(lk) do
                        done[] += 1
                        el = time() - t0
                        @printf("  2RDM %4d/%d  clusters (%s)  %.1f s  elapsed %s  left ~%s\n", done[], length(combos),
                                join([c.idx for c in combos[i]], ","), time() - ti, _hms(el),
                                _hms(el / done[] * (length(combos) - done[])))
                        flush(stdout)
                    end
                end
            end
            Gt
        end
    end
    return permutedims(sum(fetch.(tasks)), (2, 3, 4, 5, 1))
end

_hms(t) = @sprintf("%d:%02d:%02d", floor(Int, t / 3600), floor(Int, t / 60) % 60, floor(Int, t) % 60)

"""
    oncluster_2rdm!(G, v, cluster_bases, offsets; wmin=1e-14)

Adds the on-cluster 2RDM blocks (all four indices on one cluster) of each root to G (n, n, n, n, R): the cluster
density matrix rho_uv of each Fock sector, diagonalized into natural cluster states phi_k with weights w_k,
sum_uv rho_uv <u|p'q'sr|v> = sum_k w_k <phi_k|p'q'sr|phi_k>
"""
function oncluster_2rdm!(G, v::TPSChem.TPSCIstate{T,N,R}, bases, offsets; wmin=1e-14) where {T,N,R}
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
                        G[off+p, off+q, off+rr, off+s, r] += wk * (aa[p, rr, q, s] + bb[p, rr, q, s] +
                                                                   ab[p, rr, q, s] + ab[q, s, p, rr])
                    end
                end
            end
        end
    end
    return G
end
