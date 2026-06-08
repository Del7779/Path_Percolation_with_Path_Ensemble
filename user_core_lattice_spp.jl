# Infinite-horizon stochastic path attack with random OD sampling (excluding isolated nodes)
# and preferential/noisy walk path selection (NO parents-based backtracking).
#
# Assumptions about your existing state (SPPState_disorder-like):
#   st.adj[u]      :: Vector{Tuple{Int,Int}}   # (neighbor, edge_id)
#   st.removed[e]  :: Bool                    # whether edge is removed
#   st.dist[u]     :: Int                     # BFS distances (will be overwritten)
#   st.queue       :: Vector{Int}             # BFS queue buffer (length >= nv)
#   st.seen        :: Vector{Int}             # nodes touched in last BFS (for fast reset)
#   st.x[e]        :: Real                    # per-edge disorder/cost used in exp(-x/T_x)
#
# You provide and maintain:
#   deg[u] :: Int   # remaining degree with respect to unremoved edges

## Set up container
struct SPPState
    dist::Vector{Int}
    σ::Vector{Float64}
    parents::Vector{Vector{Int}}
    parents_idx::Vector{Vector{Int}}
    queue::Vector{Int}
    adj::Vector{Vector{Tuple{Int,Int}}}
    removed::BitVector
    seen::Vector{Int}
    deg::Vector{Int}
end
"""
    init_sppstate!(st::SPPState, g::SimpleGraph,x_t::Vector{Float64},s_t::Vector{Float64})

Resets `st` for graph `g` (keeps buffers, clears contents).
"""
function init_sppstate!(st::SPPState, g::SimpleGraph)
    n = nv(g)
    st.removed .= false
    empty!(st.seen)
    for i in 1:n
        st.dist[i] = typemax(Int)
        st.σ[i]    = 0
        empty!(st.parents[i])
        empty!(st.parents_idx[i])
    end
    st.deg .= degree(g)
    return st
end

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

using Random
using StatsBase

"""ActivePool maintains the set of nodes with current remaining-degree > 0."""
struct ActivePool
    nodes::Vector{Int}   # active node ids
    pos::Vector{Int}     # pos[u] = index of u in nodes, or 0 if inactive
end

"""Initialize remaining-degree vector and ActivePool from an initial Graphs.SimpleGraph."""
function init_active_pool(g)
    n = nv(g)
    deg = Vector{Int}(undef, n)
    nodes = Int[]
    pos = zeros(Int, n)
    for u in 1:n
        du = degree(g, u)
        deg[u] = du
        if du > 0
            push!(nodes, u)
            pos[u] = length(nodes)
        end
    end
    return deg, ActivePool(nodes, pos)
end

@inline function deactivate!(pool::ActivePool, u::Int)
    i = pool.pos[u]
    if i == 0
        return
    end
    last = pool.nodes[end]
    pool.nodes[i] = last
    pool.pos[last] = i
    pop!(pool.nodes)
    pool.pos[u] = 0
end

"""Sample two distinct active nodes uniformly from the active pool."""
@inline function sample_active_pair(pool::ActivePool)
    m = length(pool.nodes)
    @assert m ≥ 2
    i = rand(1:m)
    j = rand(1:m-1)
    if j ≥ i
        j += 1
    end
    return pool.nodes[i], pool.nodes[j]
end

function sample_shortest_path!(o::Int, d::Int, C::Int, st::SPPState; weighted::Bool=true)
    # clear only nodes touched last time
    for u in st.seen
        st.dist[u] = typemax(Int)
        st.σ[u]    = 0
        empty!(st.parents[u])
        empty!(st.parents_idx[u])
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
                push!(st.parents_idx[v], eid)
                push!(st.parents[v], u)
                push!(st.seen, v)
                tail += 1
                st.queue[tail] = v
                if v == d; target = st.dist[v]; end
            elseif st.dist[v] == st.dist[u] + 1
                st.σ[v] += st.σ[u]
                push!(st.parents[v], u)
                push!(st.parents_idx[v], eid)
            end

        end
    end

    # no path ≤ C
    if st.dist[d] == typemax(Int) || st.dist[d] > C
        return (nothing,nothing)
    end

    # backtrack a shortest path
    L = st.dist[d]
    path = Vector{Int}(undef, L+1)
    path_eids = Vector{Int}(undef, L)
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
        eids = st.parents_idx[cur]
        if weighted
            ws = st.σ[ps]
            k = sample(eachindex(ps), StatsBase.weights(ws))
        else
            k = sample(eachindex(ps))
        end
        path_eids[idx] = eids[k]
        cur = ps[k]
    end
    return path, path_eids
end


"""Remove all edges along a node-path and update deg + active pool incrementally.

`path` is a node list [d, ..., o] (or reversed) with consecutive nodes adjacent.
Returns number of newly removed edges.
"""
function remove_path_edges!(path::Vector{Int},
                            path_eids::Vector{Int},
                            st, deg::Vector{Int}, pool::ActivePool)

    removed_now = 0
    elist = Tuple{Int,Int}[]

    for i in eachindex(path_eids)
        eid = path_eids[i]

        if st.removed[eid]
            continue
        end

        st.removed[eid] = true
        removed_now += 1

        u = path[i]
        v = path[i+1]

        u2, v2 = minmax(u, v)
        push!(elist, (u2, v2))

        deg[u] -= 1
        deg[v] -= 1

        if deg[u] == 0
            deactivate!(pool, u)
        end
        if deg[v] == 0
            deactivate!(pool, v)
        end
    end

    return elist, removed_now
end

"""Main routine: Infinite-horizon stochastic path attack.

- Samples random OD pairs from active nodes only (deg>0).
- Runs BFS from o (infinite horizon) on remaining edges.
- Builds a preferential/noisy path from d to o (NO parents-based backtracking).
- Removes all edges on that path.
- Repeats until the network has no edges.

Returns the edge list of removals and the number of steps taken.
"""
function shortest_path_percolation!(g, st)

    time_start = time() 
    init_sppstate!(st, g);
    deg, pool = init_active_pool(g);
    remaining_edges = ne(g)
    steps = 0
    edge_logs   = Vector{Vector{Tuple{Int,Int}}}()
    while remaining_edges > 0
        if length(pool.nodes) < 2
            break
        end

        o, d = sample_active_pair(pool)

        path, path_eids = sample_shortest_path!(o, d, typemax(Int), st; weighted=true)
        if path === nothing
            continue
        end

        elist, removed_now = remove_path_edges!(path, path_eids, st, deg, pool)
        push!(edge_logs, elist)

        remaining_edges -= removed_now
        steps += 1
    end
    time_end = time()
    elapsed = time_end - time_start
    println("Elapsed time: $elapsed seconds")
    return edge_logs
end


@inline mod1_periodic(a, L) = ((a - 1) % L) + 1

@inline function cube_index(x::Int, y::Int, z::Int, L::Int)
    return x + (y - 1) * L + (z - 1) * L * L
end

function periodic_cube_graph(L::Int)
    N = L^3
    g = SimpleGraph(N)

    for z in 1:L, y in 1:L, x in 1:L
        u = cube_index(x, y, z, L)

        xp = mod1_periodic(x + 1, L)
        yp = mod1_periodic(y + 1, L)
        zp = mod1_periodic(z + 1, L)

        add_edge!(g, u, cube_index(xp, y,  z,  L))
        add_edge!(g, u, cube_index(x,  yp, z,  L))
        add_edge!(g, u, cube_index(x,  y,  zp, L))
    end

    return g
end