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

struct SPPState
    dist::Vector{Int}
    σ::Vector{Float64}
    parents::Vector{Vector{Int}}
    queue::Vector{Int}
    adj::Vector{Vector{Tuple{Int,Int}}}
    removed::BitVector
    seen::Vector{Int}
end

"""
    init_sppstate!(st::SPPState, g::SimpleGraph,x_t::Vector{Float64},s_t::Vector{Float64})

Resets `st` for graph `g` (keeps buffers, clears contents).
"""
function init_sppstate_disorder!(st::SPPState_disorder, deg::Vector, x_t::Vector{Float64},s_t::Vector{Float64})
    n = length(deg)
    st.removed .= false
    empty!(st.seen)
    for p in st.parents
        empty!(p)
    end
    for i in 1:n
        st.dist[i] = typemax(Int)
        st.σ[i]    = 0
    end
    st.x .= x_t  # new transport costs
    st.s .= s_t # new strengths
    st.deg .= deg
    return st
end

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
function check_od_withinC!(o::Int, d::Int, C::Int, st::SPPState_disorder)
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
# Noisy path sampler (PP)
# ------------------------------------

"""
    sample_noisy_path_disorder!(o, d, C, st; T_x) -> Vector{Int} | nothing

Like `sample_shortest_path!` but allows detours while descending toward `o`,
sampling next step among neighbors with non-increasing distance (and not
already visited). If `weighted`, uses σ-based weights; else uniform.
Returns `nothing` if no path ≤ C exists.
"""
function sample_noisy_path_disorder!(o::Int, d::Int, C::Int, st::SPPState_disorder; T_x::Float64=1.0,α::Float64= -1.0)
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
        return nothing, nothing
    end

    # stochastic descent from d toward o (avoid revisits)
    path = Int[]
    path_eids = Int[]
    cur  = d
    push!(path, cur)
    visited = Set{Int}([cur])

    while cur != o
        L = st.dist[cur]
        nbrs_weight = Float64[]
        nbrs = Int[]
        eids = Int[]
        for (u, eid) in st.adj[cur]
            if !st.removed[eid] && st.dist[u] ≤ L && !(u in visited)
                push!(nbrs, u)
                push!(eids, eid)
                dist_u = st.dist[u]
                # push!(nbrs_weight, exp(-st.x[eid]/T_x) * (st.deg[u]*st.deg[cur])^α)  # transport cost weigh
                push!(nbrs_weight, (st.deg[u]*st.deg[cur])^α * exp((L-dist_u)/T_x))  # transport cost weight with log for stability
            end
        end
        if isempty(nbrs)
            # fallback: give up (could also return the strict path_s)
            return nothing,nothing
        end
        idx = sample(1:length(nbrs), StatsBase.weights(nbrs_weight))
        cur = nbrs[idx]
        c = eids[idx]
        push!(path, cur)
        push!(path_eids, c)
        push!(visited, cur)
    end
    return path, path_eids
end

"""
    sample_shortest_path!(o, d, C, st; weighted=true) -> Vector{Int} | nothing

BFS up to radius C on the masked graph; backtracks a random shortest path
from `d` to `o`. If `weighted`, chooses predecessor with probability ∝ σ.
Returns `nothing` if no path of length ≤ C exists.
"""
function sample_shortest_path!(o::Int, d::Int, C::Int, st::SPPState_disorder; weighted::Bool=true)
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
        if idx == 0
            error("Backtracking error: idx reached 0 before hitting origin")
        end 
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
# Single PP / SPP run → edge logs
# ------------------------------------

"""
    run_single_spp(N, C, OD_pairs, st; weightedflag=true)
Returns: (taus, edge_logs, path_lengths)
"""
function run_single_spp(
    N::Int,
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
            # push!(edge_logs, Tuple{Int,Int}[])
            continue
        end

        # record & mask edges along path
        unique_len = length(unique(path)) - 1
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
    run_single_pp(N, C, OD_pairs, st; T_x=1.0, T_s=1.0, α=-1.0)
Returns: (taus, edge_logs, path_lengths)
"""
function run_single_pp_disorder(
    N::Int,
    C::Int,
    OD_pairs::Vector{Tuple{Int,Int}},
    st::SPPState_disorder; T_x::Float64=1.0,T_s::Float64=1.0,α::Float64=-1.0)

    od = copy(OD_pairs)
    τs = Int[]
    walk_logs   = Vector{Vector{Tuple{Int,Int}}}()
    edge_logs   = Vector{Vector{Tuple{Int,Int}}}()
    deg_width_logs = Vector{Float64}()
    path_lens   = Int64[]

    while !isempty(od)
        L = length(od)
        push!(τs, sample_tau(L, N))
        i = rand(1:L)
        o, d = od[i]

        path,path_eids = sample_noisy_path_disorder!(o, d, C, st; T_x=T_x, α=α)
        if path === nothing
            od[i] = od[end]; pop!(od)
            # push!(walk_logs, Tuple{Int,Int}[])
            # push!(edge_logs, Tuple{Int,Int}[])
            continue
        end

        unique_len = length(unique(path)) - 1
        push!(path_lens, max(unique_len, 0))
        wlist = Tuple{Int,Int}[]
        elist = Tuple{Int,Int}[]
        for j in 1:unique_len
            u, v = minmax(path[j], path[j+1])
            push!(wlist, (u, v))
            eid = path_eids[j]
            if rand() < exp(-st.s[eid]/T_s)  # strength-based removal
                st.removed[eid] = true
                push!(elist, (u, v))
                # dynamic degree update (edge removed)
                st.deg[u] -= 1
                st.deg[v] -= 1
            end
        end
        push!(deg_width_logs, maximum(st.deg)-minimum(st.deg))
        push!(edge_logs, elist)
        push!(walk_logs, wlist)

        # remove OD if no path remains
        if check_od_withinC!(o, d, C, st) === false
            od[i] = od[end]; pop!(od)
        end
    end
    
    return τs, walk_logs,edge_logs, path_lens, deg_width_logs
end


"""
    run_single_pp(g, C, OD_pairs, st; weightflag=true)
Returns: (taus, edge_logs, path_lengths)
"""
function run_single_pp(
    N,
    C::Int,
    OD_pairs::Vector{Tuple{Int,Int}},
    st::Union{SPPState, SPPState_disorder}; weightedflag::Bool=false)

    od = copy(OD_pairs)
    τs = Int[]
    edge_logs   = Vector{Vector{Tuple{Int,Int}}}()
    path_lens   = Int64[]

    while !isempty(od)
        L = length(od)
        push!(τs, sample_tau(L, N))
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

    return τs, edge_logs, path_lens
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

function holme_kim(n::Int, m::Int, p::Float64; seed=nothing)
    @assert n ≥ m + 1 "n must be > m"
    @assert 0.0 ≤ p ≤ 1.0 "p must be in [0,1]"

    if seed !== nothing
        Random.seed!(seed)
    end

    # Initial seed graph: clique of m nodes
    g = SimpleGraph(m)
    for u in 1:m, v in (u+1):m
        add_edge!(g, u, v)
    end

    # Grow network
    for new_node in (m+1):n
        add_vertex!(g)

        targets = Set{Int}()

        # ---- First edge: preferential attachment ----
        degs = degree(g)[1:new_node-1]
        first = sample(1:new_node-1, Weights(degs))
        add_edge!(g, new_node, first)
        push!(targets, first)

        last_attached = first

        # ---- Remaining m-1 edges ----
        while length(targets) < m
            if rand() < p
                # Try triad formation
                neigh = neighbors(g, last_attached)
                # Only consider old nodes not already connected
                candidates = filter(x ->
                    x < new_node && !(x in targets),
                    neigh
                )

                if !isempty(candidates)
                    v = rand(candidates)
                    add_edge!(g, new_node, v)
                    push!(targets, v)
                    last_attached = v
                    continue
                end
                # If no valid neighbor → fallback to PA
            end

            # Preferential attachment fallback
            degs = degree(g)[1:new_node-1]
            candidates = setdiff(1:new_node-1, collect(targets))
            v = sample(candidates, Weights(degs[candidates]))
            add_edge!(g, new_node, v)
            push!(targets, v)
            last_attached = v
        end
    end

    return g
end


# estimate the average distance by sampling node pairs and computing their shortest path lengths
function average_distance_sample(g::SimpleGraph; nsamples=min(1000, nv(g)))  # default to 1000 or n, whichever is larger
    n = nv(g)
    total = 0.0
    count = 0
    giant_component = connected_components(g)[1]  # get the largest connected component
    for _ in 1:nsamples
        s = rand(giant_component)
        d = dijkstra_shortest_paths(g, s).dists
        for t in 1:n
            if t != s && isfinite(d[t])
                total += d[t]
                count += 1
            end
        end
    end

    return total / count
end


"""
    edge_betweenness_C(adj, n, C; removed=nothing) -> Vector{Float64}

C-hop (distance ≤ C) edge betweenness / load on an unweighted graph,
computed by truncated Brandes.

Inputs:
- adj::Vector{Vector{Tuple{Int,Int}}}: indexed adjacency, adj[u] = [(v,eid), ...]
- n::Int: number of nodes
- C::Int: hop cutoff
- removed::Union{Nothing,BitVector}: optional edge mask; if provided, skip removed[eid]==true

Output:
- EBC::Vector{Float64} of length = maximum eid (typically ne(g)), unnormalized.
"""
function edge_betweenness_C(adj, n::Int, C::Int; removed::Union{Nothing,BitVector}=nothing)
    m = 0
    for u in 1:n
        for (_, eid) in adj[u]
            m = max(m, eid)
        end
    end
    EBC = zeros(Float64, m)

    dist    = fill(typemax(Int), n)
    sigma   = zeros(Float64, n)
    delta   = zeros(Float64, n)
    P       = [Tuple{Int,Int}[] for _ in 1:n]   # predecessors as (p, eid(p->v))
    Q       = Vector{Int}(undef, n)
    S       = Int[]                             # BFS order stack
    touched = Int[]                             # nodes reached in this BFS

    for s in 1:n
        # reset only touched nodes
        for v in touched
            dist[v]  = typemax(Int)
            sigma[v] = 0.0
            delta[v] = 0.0
            empty!(P[v])
        end
        empty!(touched)
        empty!(S)

        # BFS from s up to depth C
        head = 1; tail = 1
        Q[1] = s
        dist[s]  = 0
        sigma[s] = 1.0
        push!(touched, s)

        while head ≤ tail
            v = Q[head]; head += 1
            push!(S, v)

            dv = dist[v]
            if dv == C
                continue
            end

            @inbounds for (w, eid) in adj[v]
                if removed !== nothing && removed[eid]
                    continue
                end

                if dist[w] == typemax(Int)
                    dist[w] = dv + 1
                    tail += 1
                    Q[tail] = w
                    push!(touched, w)
                end

                if dist[w] == dv + 1
                    sigma[w] += sigma[v]
                    push!(P[w], (v, eid))  # v is predecessor of w via edge eid
                end
            end
        end

        # Back-propagation of dependencies (reverse BFS order)
        for w in reverse(S)
            @inbounds for (v, eid) in P[w]
                # share of shortest-path flow from s that goes through (v,w)
                c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                delta[v] += c
                EBC[eid] += c
            end
        end
    end

    # undirected correction
    EBC ./= 2.0
    return EBC
end

function holme_kim_fast(n::Int, m::Int, p::Float64; seed=nothing)
    @assert n >= m + 1 "n must be >= m + 1"
    @assert 0.0 <= p <= 1.0 "p must be in [0,1]"

    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    g = Graphs.SimpleGraph(n)

    # Degree bookkeeping for O(1) preferential attachment
    deg = zeros(Int32, n)

    # Exact number of undirected edges in a Holme-Kim graph with these params
    total_edges = (m * (m - 1)) ÷ 2 + m * (n - m)
    stubs = Vector{Int32}(undef, 2 * total_edges)
    stub_count = 0

    @inline function push_stub!(v::Int)
        stub_count += 1
        stubs[stub_count] = Int32(v)
        return nothing
    end

    @inline function add_edge_fast!(u::Int, v::Int)
        Graphs.add_edge!(g, u, v)

        deg[u] += 1
        push_stub!(u)

        deg[v] += 1
        push_stub!(v)

        return nothing
    end

    # Initial clique on 1:m
    for u in 1:(m - 1), v in (u + 1):m
        add_edge_fast!(u, v)
    end

    targets = falses(n)              # nodes already connected to current new node
    picked  = Vector{Int}(undef, m)   # reused buffer
    last_attached = 0

    # Growth phase
    for new_node in (m + 1):n
        len = 0

        # First edge: preferential attachment
        while true
            v = Int(stubs[rand(rng, 1:stub_count)])
            if !targets[v]
                add_edge_fast!(new_node, v)
                targets[v] = true
                len += 1
                picked[len] = v
                last_attached = v
                break
            end
        end

        # Remaining m - 1 edges
        while len < m
            v = 0

            # Triad formation attempt
            if rand(rng) < p
                seen = 0
                chosen = 0
                for w in Graphs.neighbors(g, last_attached)
                    if w < new_node && !targets[w]
                        seen += 1
                        if rand(rng, 1:seen) == 1
                            chosen = w
                        end
                    end
                end
                v = chosen
            end

            # Fallback to preferential attachment
            if v == 0
                while true
                    v = Int(stubs[rand(rng, 1:stub_count)])
                    if !targets[v]
                        break
                    end
                end
            end

            add_edge_fast!(new_node, v)
            targets[v] = true
            len += 1
            picked[len] = v
            last_attached = v
        end

        @inbounds for i in 1:len
            targets[picked[i]] = false
        end
    end

    return g
end