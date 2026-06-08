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

using Random
using StatsBase

"""ActivePool maintains the set of nodes with current remaining-degree > 0."""
struct ActivePool
    nodes::Vector{Int}   # active node ids
    pos::Vector{Int}     # pos[u] = index of u in nodes, or 0 if inactive
end

"""Initialize remaining-degree vector and ActivePool from an initial Graphs.SimpleGraph."""
function init_active_pool(n)
    nodes = collect(1:n)
    pos = collect(1:n)
    return ActivePool(nodes, pos)
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

"""Reset only nodes that were touched in the previous BFS."""
@inline function _reset_bfs!(st)
    for u in st.seen
        st.dist[u] = typemax(Int)
    end
    empty!(st.seen)
end

"""Infinite-horizon BFS from origin o to destination d with constraint C. Fills st.dist."""
function bfs_to!(o::Int, d::Int, st, C)
    _reset_bfs!(st)

    head = 1
    tail = 1
    st.queue[1] = o
    st.dist[o] = 0
    push!(st.seen, o)

    target_dist = typemax(Int)

    while head ≤ tail
        u = st.queue[head]; head += 1

        if st.dist[u] >= C
            continue
        end

        # Once we've moved past d's shell, every useful distance is settled
        if st.dist[u] > target_dist
            return true
        end

        du = st.dist[u]
        @inbounds for (v, eid) in st.adj[u]
            if st.removed[eid]
                continue
            end
            if st.dist[v] == typemax(Int)
                st.dist[v] = du + 1
                push!(st.seen, v)
                tail += 1
                st.queue[tail] = v

                # Record d's shell depth the first time d is enqueued
                if v == d
                    target_dist = du + 1
                end
            end
        end
    end

    return target_dist != typemax(Int)
end

"""finite-horizon BFS from origin o to destination d. Fills st.dist."""
function bfs_to_finite!(o::Int, st,C)
    _reset_bfs!(st)

    head = 1
    tail = 1
    st.queue[1] = o
    st.dist[o] = 0
    push!(st.seen, o)

    while head ≤ tail
        u = st.queue[head]; head += 1

        du = st.dist[u]
        if du >= C
            continue
        end

        @inbounds for (v, eid) in st.adj[u]
            if st.removed[eid]
                continue
            end
            if st.dist[v] == typemax(Int)
                st.dist[v] = du + 1
                push!(st.seen, v)
                tail += 1
                st.queue[tail] = v
            end
        end
    end
    return true
end

function sample_preferential_path_finite!(o::Int, st; T_x::Real=Inf, α::Real=0.0,d::Int = sample(st.seen),
                                       use_dist_constraint::Bool=true,
                                       max_steps::Int=0)
    dist_d = st.dist[d]
    if dist_d == typemax(Int)
        return nothing
    end

    if max_steps <= 0
        max_steps = length(st.dist)
    end

    cur = d
    path = Int[cur]
    path_eids = Int[]   # 

    visited = path

    steps = 0
    while cur != o
        steps += 1
        if steps > max_steps
            return nothing
        end

        L = st.dist[cur]

        nbrs = Int[]
        eids = Int[]      # 
        wts  = Float64[]

        @inbounds for (v, eid) in st.adj[cur]
            if st.removed[eid]
                continue
            end

            dv_dist = st.dist[v]
            if dv_dist == typemax(Int)
                continue
            end
            if use_dist_constraint && dv_dist > L
                continue
            end

            # visited check
            found = false
            for p in visited
                if p == v
                    found = true
                    break
                end
            end
            found && continue

            dv = max(st.deg[v], 1)
            dc = max(st.deg[cur], 1)
            dist_difference = L - dv_dist

            w = (float(dv * dc))^float(α) * exp(float(dist_difference) / float(T_x))

            if isfinite(w) && w > 0
                push!(nbrs, v)
                push!(eids, eid)   # 
                push!(wts, w)
            end
        end

        if isempty(nbrs)
            return nothing
        end

        idx = sample(1:length(nbrs), Weights(wts))
        cur = nbrs[idx]

        push!(path, cur)
        push!(path_eids, eids[idx])   # 
    end

    return path, path_eids   # 
end

"""Preferential/noisy walk from d toward o using local edge/node weights.

Weights are proportional to:
    exp(-x_e / T_x) * (deg[next] * deg[cur])^α

If use_dist_constraint=true, only considers neighbors with dist[next] ≤ dist[cur]
(where dist comes from BFS rooted at o). This biases motion toward o while
still allowing lateral detours.

Returns a vector of node ids [d, ..., o] or `nothing` if it gets stuck.
"""
function sample_preferential_path_inf!(o::Int, d::Int, st;
                                       T_x::Real=Inf,
                                       α::Real=0.0,
                                       use_dist_constraint::Bool=true,
                                       max_steps::Int=0)

    dist_d = st.dist[d]
    if dist_d == typemax(Int)
        return nothing
    end

    if max_steps <= 0
        max_steps = length(st.dist)
    end

    cur = d
    path = Int[cur]
    path_eids = Int[]   # 

    visited = path

    steps = 0
    while cur != o
        steps += 1
        if steps > max_steps
            return nothing
        end

        L = st.dist[cur]

        nbrs = Int[]
        eids = Int[]      # 
        wts  = Float64[]

        @inbounds for (v, eid) in st.adj[cur]
            if st.removed[eid]
                continue
            end

            dv_dist = st.dist[v]
            if dv_dist == typemax(Int)
                continue
            end
            if use_dist_constraint && dv_dist > L
                continue
            end

            # visited check
            found = false
            for p in visited
                if p == v
                    found = true
                    break
                end
            end
            found && continue

            dv = max(st.deg[v], 1)
            dc = max(st.deg[cur], 1)
            dist_difference = L - dv_dist

            w = (float(dv * dc))^float(α) * exp(float(dist_difference) / float(T_x))

            if isfinite(w) && w > 0
                push!(nbrs, v)
                push!(eids, eid)   # 
                push!(wts, w)
            end
        end

        if isempty(nbrs)
            return nothing
        end

        idx = sample(1:length(nbrs), Weights(wts))
        cur = nbrs[idx]

        push!(path, cur)
        push!(path_eids, eids[idx])   # 
    end

    return path, path_eids   # 
end

"""Remove all edges along a node-path and update deg + active pool incrementally.

`path` is a node list [d, ..., o] (or reversed) with consecutive nodes adjacent.
Returns number of newly removed edges.
"""
function remove_path_edges!(path::Vector{Int},
                            path_eids::Vector{Int},
                            st, pool::ActivePool;
                            T_s::Real=Inf)

    removed_now = 0
    wlist = Tuple{Int,Int}[]
    elist = Tuple{Int,Int}[]

    for i in eachindex(path_eids)
        eid = path_eids[i]

        if st.removed[eid]
            continue
        end

        if rand() > exp(-float(st.s[eid]) / float(T_s))
            continue
        end

        st.removed[eid] = true
        removed_now += 1

        u = path[i]
        v = path[i+1]

        u2, v2 = minmax(u, v)
        push!(wlist, (u2, v2))
        push!(elist, (u2, v2))

        st.deg[u] -= 1
        st.deg[v] -= 1

        if st.deg[u] == 0
            deactivate!(pool, u)
        end
        if st.deg[v] == 0
            deactivate!(pool, v)
        end
    end
    deg_width = maximum(st.deg) - minimum(st.deg)

    return wlist, elist, deg_width, removed_now
end

"""Main routine: Infinite-horizon stochastic path attack.

- Samples random OD pairs from active nodes only (deg>0).
- Runs BFS from o (infinite horizon) on remaining edges.
- Builds a preferential/noisy path from d to o (NO parents-based backtracking).
- Removes all edges on that path.
- Repeats until the network has no edges.

Returns the edge list of removals and the number of steps taken.
"""
function random_path_percolation_infinite_preferential!(deg,st,C;
        x_t::Vector{Float64},
        s_t::Vector{Float64},
        T_x::Real= Inf,
        T_s::Real = Inf,
        α::Real=-1.0,
        use_dist_constraint::Bool=true,
        max_steps_walk::Int=0,
        max_trials::Int=0,
        pair_uniform::Bool=false)

    time_start = time() 
    n = length(deg)
    init_sppstate_disorder!(st, deg, x_t, s_t)   
    pool = init_active_pool(n)
    remaining_edges = sum(deg) ÷ 2
    steps = 0

    # Optional: prevent infinite looping if graph becomes highly disconnected
    # max_trials=0 means unlimited
    trials = 0
    walk_logs   = Vector{Vector{Tuple{Int,Int}}}()
    edge_logs   = Vector{Vector{Tuple{Int,Int}}}()
    deg_width_log = Int[]
    while remaining_edges > 0

        if max_trials > 0
            trials += 1
            if trials > max_trials
                break
            end
        end

        o, d = sample_active_pair(pool)

        if !bfs_to!(o, d, st, C)
            if  pair_uniform
                continue
            end
            d = sample(st.seen)  # fallback to random reachable node if d is unreachable
        end

        path, path_eids = sample_preferential_path_inf!(o, d, st;
                    T_x= T_x, α=α,
                    use_dist_constraint=use_dist_constraint,
                    max_steps=max_steps_walk)
        
        if path === nothing
            continue
        end

        wlist, elist, deg_width, removed_now = remove_path_edges!(path, path_eids, st, pool; T_s=T_s)
        push!(walk_logs, wlist)
        push!(edge_logs, elist)
        push!(deg_width_log, deg_width)

        remaining_edges -= removed_now
        steps += 1
    end
    time_end = time()
    elapsed = time_end - time_start
    # println("Elapsed time: $elapsed seconds")
    return walk_logs, edge_logs, deg_width_log
end


"""Main routine: finite-horizon stochastic path attack.

- Samples random OD pairs from active nodes only (deg>0).
- Runs BFS from o (finite horizon) on remaining edges and cut at C.
- Builds a preferential/noisy path from d to o (backtracking).
- Removes all edges on that path.
- Repeats until the network has no edges.

Returns the edge list of removals and the number of steps taken.
"""
function random_path_percolation_finite_preferential!(deg, st, C;
        x_t::Vector{Float64},
        s_t::Vector{Float64},
        T_x::Real= Inf,
        T_s::Real = Inf,
        α::Real=-1.0,
        use_dist_constraint::Bool=true,
        max_steps_walk::Int=0,
        max_trials::Int=0)

    time_start = time() 
    n = length(deg)
    init_sppstate_disorder!(st, deg, x_t, s_t)   
    pool = init_active_pool(n)
    remaining_edges = sum(deg) ÷ 2
    steps = 0

    # Optional: prevent infinite looping if graph becomes highly disconnected
    # max_trials=0 means unlimited
    trials = 0
    walk_logs   = Vector{Vector{Tuple{Int,Int}}}()
    edge_logs   = Vector{Vector{Tuple{Int,Int}}}()
    deg_width_log = []
    while remaining_edges > 0

        if max_trials > 0
            trials += 1
            if trials > max_trials
                break
            end
        end
        
        o = sample(pool.nodes)

        if !bfs_to_finite!(o, st,C)
            continue
        end

        path, path_eids = sample_preferential_path_finite!(o, st;
                    T_x= T_x, α=α,
                    use_dist_constraint=use_dist_constraint,
                    max_steps=max_steps_walk)
        if path === nothing
            continue
        end

        wlist, elist, deg_width, removed_now = remove_path_edges!(path, path_eids, st, pool; T_s=T_s)
        push!(walk_logs, wlist)
        push!(edge_logs, elist)
        push!(deg_width_log, deg_width)

        remaining_edges -= removed_now
        steps += 1
    end
    time_end = time()
    elapsed = time_end - time_start
    # println("Elapsed time: $elapsed seconds")
    return walk_logs, edge_logs, deg_width_log
end
