# user_core.jl
# Core routines for path-based percolation experiments (PP & SPP).
# Include this file on every worker:  @everywhere include("user_core.jl")

using Random
using Statistics
using Graphs
using Distributions
using StatsBase

# Make sure Newman_Ziff.jl is alongside this file
include("Newman_Ziff.jl")
# using .Newman_Ziff

# ---------------------------
# Helpers for OD pairs & BFS
# ---------------------------

"""
    find_node_pairs_within_distance(g::SimpleGraph, C::Int) -> Vector{Tuple{Int,Int}}

All unordered node pairs (u,v), u<v, whose graph distance is ≤ C.
"""
function find_node_pairs_within_distance(g::SimpleGraph, C::Int)
    n = nv(g)
    pairs = Vector{Tuple{Int,Int}}()

    seen  = falses(n)
    dist  = Vector{Int}(undef, n)
    queue = Vector{Int}(undef, n + 1)

    for u in 1:n
        fill!(seen, false)
        head = 1; tail = 1
        queue[1] = u
        seen[u] = true
        dist[u] = 0

        while head ≤ tail
            v = queue[head]; head += 1
            d = dist[v]
            if d == C
                continue
            end
            @inbounds for w in neighbors(g, v)
                if !seen[w]
                    seen[w] = true
                    dist[w] = d + 1
                    tail += 1
                    queue[tail] = w
                    if u < w
                        push!(pairs, (u, w))
                    end
                end
            end
        end
    end
    return pairs
end

"""
    sample_tau(L::Int, N::Int) -> Int

Geometric draw with parameter p = 2L / (N(N-1)), as in your process.
"""
function sample_tau(L::Int, N::Int)
    p = 2L / (N * (N - 1))
    return rand(Geometric(p))  # τ ∈ {1,2,3,...}
end

"""
    build_indexed_adjacency(g) -> Vector{Vector{Tuple{Int,Int}}}

Adjacency with edge IDs: adj[u] = [(v,eid), ...] with eid ∈ 1:ne(g).
"""
function build_indexed_adjacency(g::SimpleGraph)
    n   = nv(g)
    adj = [Vector{Tuple{Int,Int}}() for _ in 1:n]
    for (eid, e) in enumerate(edges(g))
        u, v = src(e), dst(e)
        push!(adj[u], (v, eid))
        push!(adj[v], (u, eid))
    end
    return adj
end

# ---------------------------
# State for repeated BFS
# ---------------------------

"""
Container reused across BFS/backtracking calls to avoid allocations.
"""
struct SPPState
    dist::Vector{Int}                       # distances
    σ::Vector{Int}                          # #shortest paths to node
    parents::Vector{Vector{Int}}            # predecessor list per node
    queue::Vector{Int}                      # BFS queue
    adj::Vector{Vector{Tuple{Int,Int}}}     # indexed adjacency (v,eid)
    removed::BitVector                      # masked edges
    seen::Vector{Int}                       # nodes touched in last run
end

struct SPPState_disorder
    dist::Vector{Int}
    σ::Vector{Float64}
    parents::Vector{Vector{Int}}
    queue::Vector{Int}
    adj::Vector{Vector{Tuple{Int,Int}}}
    removed::BitVector
    seen::Vector{Int}
    x::Vector{Float64}   # transport cost
    s::Vector{Float64}   # strength
    deg::Vector{Float64}   # disorder parameter (optional)
end


"""
    check_od_withinC!(o, d, C, st; weighted=true) -> Bool

BFS up to radius C on the masked graph; returns true if d is reached. 
"""
function check_od_withinC!(o::Int, d::Int, C::Int, st::Union{SPPState, SPPState_disorder})
    # reset touched nodes
    for u in st.seen
        st.dist[u] = typemax(Int)
        st.σ[u]    = 0
        empty!(st.parents[u])
    end
    empty!(st.seen)

    head = 1; tail = 1
    st.queue[1] = o
    st.dist[o] = 0
    push!(st.seen, o)

    while head ≤ tail
        u = st.queue[head]; head += 1
        if u == d
            return true
        end
        if st.dist[u] == C
            continue
        end
        @inbounds for (v, eid) in st.adj[u]
            if st.removed[eid]; continue; end
            if st.dist[v] == typemax(Int)
                st.dist[v] = st.dist[u] + 1
                push!(st.seen, v)
                tail += 1
                st.queue[tail] = v
            end
        end
    end
    return false
end
"""
    init_sppstate!(st, n)

Resets `st` for graph `g` (keeps buffers, clears contents).
"""
function init_sppstate!(st, n)
    st.removed .= false
    empty!(st.seen)
    for p in st.parents
        empty!(p)
    end
    for i in 1:n
        st.dist[i] = typemax(Int)
        st.σ[i]    = 0
    end
    return st
end

"""
    check_od_withinC!(o, d, C, st; weighted=true) -> Bool

BFS up to radius C on the masked graph; returns true if d is reached. 
"""
function check_od_withinC!(o::Int, d::Int, C::Int, st::Union{SPPState, SPPState_disorder})
    # reset touched nodes
    for u in st.seen
        st.dist[u] = typemax(Int)
        st.σ[u]    = 0
        empty!(st.parents[u])
    end
    empty!(st.seen)

    head = 1; tail = 1
    st.queue[1] = o
    st.dist[o] = 0
    push!(st.seen, o)

    while head ≤ tail
        u = st.queue[head]; head += 1
        if u == d
            return true
        end
        if st.dist[u] == C
            continue
        end
        @inbounds for (v, eid) in st.adj[u]
            if st.removed[eid]; continue; end
            if st.dist[v] == typemax(Int)
                st.dist[v] = st.dist[u] + 1
                push!(st.seen, v)
                tail += 1
                st.queue[tail] = v
            end
        end
    end
    return false
end

# ------------------------------------
# Shortest-path sampler (strict SPP)
# ------------------------------------

"""
    sample_shortest_path!(o, d, C, st; weighted=true) -> Vector{Int} | nothing

BFS up to radius C on the masked graph; backtracks a random shortest path
from `d` to `o`. If `weighted`, chooses predecessor with probability ∝ σ.
Returns `nothing` if no path of length ≤ C exists.
"""
function sample_shortest_path!(o::Int, d::Int, C::Int, st::SPPState; weighted::Bool=true)
    # clear only nodes touched last time
    for u in st.seen
        st.dist[u] = typemax(Int)
        st.σ[u]    = 0
        empty!(st.parents[u])
    end
    empty!(st.seen)

    # seed BFS
    head = 1; tail = 1
    st.queue[1] = o
    st.dist[o]  = 0
    st.σ[o]     = 1
    push!(st.seen, o)
    target = typemax(Int)

    # BFS up to C
    while head ≤ tail
        u = st.queue[head]; head += 1
        if u == d || st.dist[u] >= C || st.dist[u] > target
            continue
        end
        @inbounds for (v, eid) in st.adj[u]
            if st.removed[eid]; continue; end
            if st.dist[v] == typemax(Int)
                st.dist[v] = st.dist[u] + 1
                st.σ[v]    = st.σ[u]
                push!(st.parents[v], u)
                push!(st.seen, v)
                tail += 1
                st.queue[tail] = v
                if v == d; target = st.dist[v]; end
            elseif st.dist[v] == st.dist[u] + 1
                st.σ[v] += st.σ[u]
                push!(st.parents[v], u)
            end
        end
    end

    # no path ≤ C
    if st.dist[d] == typemax(Int) || st.dist[d] > C
        return nothing
    end

    # backtrack a shortest path
    L = st.dist[d]
    path = Vector{Int}(undef, L+1)
    idx  = L+1
    cur  = d
    while true
        path[idx] = cur
        idx -= 1
        if cur == o; break; end
        ps = st.parents[cur]
        if weighted
            ws = st.σ[ps]
            cur = sample(ps, StatsBase.weights(ws))
        else
            cur = sample(ps)
        end
    end
    return path
end

# ------------------------------------
# Noisy path sampler (PP)
# ------------------------------------

"""
    sample_noisy_path!(o, d, C, st; weighted=true) -> Vector{Int} | nothing

Like `sample_shortest_path!` but allows detours while descending toward `o`,
sampling next step among neighbors with non-increasing distance (and not
already visited). If `weighted`, uses σ-based weights; else uniform.
Returns `nothing` if no path ≤ C exists.
"""
function sample_noisy_path!(o::Int, d::Int, C::Int, st::SPPState; weighted::Bool=false)
    # clear touched nodes
    for u in st.seen
        st.dist[u] = typemax(Int)
        st.σ[u]    = 0
        empty!(st.parents[u])
    end
    empty!(st.seen)

    # seed BFS
    head = 1; tail = 1
    st.queue[1] = o
    st.dist[o]  = 0
    st.σ[o]     = 1
    push!(st.seen, o)
    target = typemax(Int)

    # BFS up to C
    while head ≤ tail
        u = st.queue[head]; head += 1
        if u == d || st.dist[u] >= C || st.dist[u] > target
            continue
        end
        @inbounds for (v, eid) in st.adj[u]
            if st.removed[eid]; continue; end
            if st.dist[v] == typemax(Int)
                st.dist[v] = st.dist[u] + 1
                st.σ[v]    = st.σ[u]
                push!(st.parents[v], u)
                push!(st.seen, v)
                tail += 1
                st.queue[tail] = v
                if v == d; target = st.dist[v]; end
            elseif st.dist[v] == st.dist[u] + 1
                st.σ[v] += st.σ[u]
                push!(st.parents[v], u)
            end
        end
    end

    # no path ≤ C
    if st.dist[d] == typemax(Int) || st.dist[d] > C
        return nothing
    end

    # stochastic descent from d toward o (avoid revisits)
    path = Int[]
    cur  = d
    push!(path, cur)
    visited = Set{Int}([cur])

    while cur != o
        L = st.dist[cur]
        nbrs = Int[]
        for (u, eid) in st.adj[cur]
            if !st.removed[eid] && st.dist[u] ≤ L && !(u in visited)
                push!(nbrs, u)
            end
        end
        if isempty(nbrs)
            # fallback: give up (could also return the strict path_s)
            return nothing
        end
        if weighted
            cur = sample(nbrs, StatsBase.weights(st.σ[nbrs]))
        else
            cur = sample(nbrs)
        end
        push!(path, cur)
        push!(visited, cur)
    end
    return path
end

# ------------------------------------
# Single PP / SPP run → edge logs
# ------------------------------------

"""
    run_single_spp(g, C, OD_pairs, st; weightedflag=true)
Returns: (taus, edge_logs, path_lengths)
"""
function run_single_spp(N,
    C::Int,
    OD_pairs::Vector{Tuple{Int,Int}},
    st::SPPState; weightedflag::Bool=true)

    od = copy(OD_pairs)
    τs = Int[]
    edge_logs   = Vector{Vector{Tuple{Int,Int}}}()
    path_lens   = Int64[]

    while !isempty(od)
        L = length(od)
        push!(τs, sample_tau(L, N))
        i = rand(1:L)
        o, d = od[i]

        path = sample_shortest_path!(o, d, C, st; weighted=weightedflag)
        if path === nothing
            od[i] = od[end]; pop!(od)
            push!(edge_logs, Tuple{Int,Int}[])
            continue
        end

        # record & mask edges along path
        unique_len = length(path) - 1
        push!(path_lens, max(unique_len, 0))
        elist = Vector{Tuple{Int,Int}}(undef, max(unique_len, 0))
        for j in 1:unique_len
            u, v = minmax(path[j], path[j+1])
            elist[j] = (u, v)
            pos = findfirst(x -> x[1] == v, st.adj[u])
            eid = st.adj[u][pos][2]
            st.removed[eid] = true
        end
        push!(edge_logs, elist)
        
        # remove OD if no path remains under mask
        if check_od_withinC!(o, d, C, st) === false
            od[i] = od[end]; pop!(od)
        end
    end

    return τs, edge_logs, path_lens
end

"""
    run_single_pp(g, C, OD_pairs, st; weightflag=true)
Returns: (taus, edge_logs, path_lengths)
"""
function run_single_pp(
    N::Int,
    C::Int,
    OD_pairs::Vector{Tuple{Int,Int}},
    st::SPPState; weightedflag::Bool=true)

    od = copy(OD_pairs)
    edge_logs   = Vector{Vector{Tuple{Int,Int}}}()
    path_lens   = Int64[]

    while !isempty(od)
        L = length(od)
        i = rand(1:L)
        o, d = od[i]

        path = sample_noisy_path!(o, d, C, st; weighted=weightedflag)
        if path === nothing
            od[i] = od[end]; pop!(od)
            push!(edge_logs, Tuple{Int,Int}[])
            continue
        end

        unique_len = length(path) - 1
        push!(path_lens, max(unique_len, 0))
        elist = Vector{Tuple{Int,Int}}(undef, max(unique_len, 0))
        for j in 1:unique_len
            u, v = minmax(path[j], path[j+1])
            elist[j] = (u, v)
            pos = findfirst(x -> x[1] == v, st.adj[u])
            eid = st.adj[u][pos][2]
            st.removed[eid] = true
        end
        push!(edge_logs, elist)

        # remove OD if no path remains under mask
        if check_od_withinC!(o, d, C, st) === false
            od[i] = od[end]; pop!(od)
        end
    end

    return edge_logs, path_lens
end

# ------------------------------------
# Convenience (optional)
# ------------------------------------

"""
    run_single_spp_newman(G, C, OD_pairs, st; points=50)

One SPP realization → feed edge sequence to Newman–Ziff,
return (p_values, s_max, chi, path_lengths).
"""
function run_single_spp_newman(
    G::SimpleGraph, C::Int, OD_pairs::Vector{Tuple{Int,Int}},
    st::SPPState; points::Int=50)

    init_sppstate!(st, G)
    _, edge_list_tau, path_lens = run_single_spp(G, C, OD_pairs, st)
    edge_list = [(x...,) for ee in edge_list_tau for x in ee]
    edge_list = reverse(edge_list)
    windows   = round.(Int, collect(range(1, ne(G)-1; length=points)))
    p_vals, s_max, chi = run_single_trial(nv(G), edge_list, windows; shuffle=false)
    return p_vals, s_max, chi, path_lens
end

# ------------------------------------
# Scale-free generator (UCM)
# ------------------------------------

"""
    ucm_scale_free(n; λ=2.5, kmin=2, seed=nothing, check_graphical=false)
→ (g::SimpleGraph, degs::Vector{Int}, info::NamedTuple)

Uncorrelated configuration model with structural cutoff kmax = ⌊√n⌋.
"""
function ucm_scale_free(n; λ::Real=2.5, kmin::Int=2,
                        seed::Union{Nothing,Int}=nothing,
                        check_graphical::Bool=false)
    @assert n ≥ 2 "n must be ≥ 2"
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    kmax = max(kmin, floor(Int, sqrt(n)))
    ks = collect(kmin:kmax)
    w  = Weights((k -> k^(-float(λ))).(ks))

    degs = sample(rng, ks, w, n, replace=true)

    # Make sum even
    if isodd(sum(degs))
        i = rand(rng, 1:n)
        if degs[i] < kmax
            degs[i] += 1
        else
            degs[i] -= 1
        end
    end

    g = random_configuration_model(n, degs; rng=rng, check_graphical=check_graphical)
    return g, degs, (kmin=kmin, kmax=kmax, λ=float(λ))
end
