using Graphs, Random, StatsBase, Statistics
include("../../Newman_Ziff.jl")
include("../../user_core_weighted.jl")
include("../../user_core_infinite_prefwalk.jl")

# ---------- Helpers ----------
function edge_entropy(counts)
    p = counts ./ sum(counts)
    return -sum(p .* log.(p .+ 1e-12))
end

function count_edges(path_eids, m)
    counts = zeros(Float64, m)
    for eids in path_eids
        for eid in eids
            counts[eid] += 1
        end
    end
    return counts
end

# ---------- Main experiment ----------
function run_entropy_experiment(; 
    N=10_000,
    k=4,
    beta=0.1,
    num_trials=50,
    sample_paths=5000
)

    g = newman_watts_strogatz(N, k, beta)
    m = ne(g)

    configs = [
        # pair_uniform
        (T=0.2,  C=3,                    mode=:pair_uniform),
        (T=Inf,  C=3,                    mode=:pair_uniform),
        (T=0.2,  C=ceil(Int,N^(1/3)),    mode=:pair_uniform),
        (T=Inf,  C=ceil(Int,N^(1/3)),    mode=:pair_uniform),

        # source_uniform
        (T=0.2,  C=3,                    mode=:source_uniform),
        (T=Inf,  C=3,                    mode=:source_uniform),
        (T=0.2,  C=ceil(Int,N^(1/3)),    mode=:source_uniform),
        (T=Inf,  C=ceil(Int,N^(1/3)),    mode=:source_uniform),

        # mixed
        (T=0.2,  C=ceil(Int,N^(1/3)),    mode=:mixed),
        (T=Inf,  C=ceil(Int,N^(1/3)),    mode=:mixed),
    ]

    results = Dict()

    for cfg in configs
        println("Running: ", cfg)

        entropies = Float64[]
        pc_vals   = Float64[]

        for trial in 1:num_trials
            deg = degree(g)
            adj = build_indexed_adjacency(g)

            st = SPPState_disorder(
                Vector{Int}(undef, N),
                Vector{Int}(undef, N),
                [Int[] for _ in 1:N],
                Vector{Int}(undef, N + 1),
                adj,
                falses(m),
                Int[],
                zeros(m),
                zeros(m),
                zeros(N),
            )

            x_t = ones(m)
            s_t = ones(m)
            init_sppstate_disorder!(st, deg, x_t, s_t)

            # ---- choose ensemble ----
            if cfg.mode == :pair_uniform
                _, edge_list_tau, _ = random_path_percolation_infinite_preferential!(
                    deg, st, cfg.C;
                    x_t=x_t, s_t=s_t, T_s=Inf, α=0.0,
                    T_x=cfg.T, pair_uniform=true
                )
            elseif cfg.mode == :source_uniform
                _, edge_list_tau, _ = random_path_percolation_finite_preferential!(
                    deg, st, cfg.C;
                    x_t=x_t, s_t=s_t, T_s=Inf, α=0.0,
                    T_x=cfg.T
                )
            else # mixed
                _, edge_list_tau, _ = random_path_percolation_infinite_preferential!(
                    deg, st, cfg.C;
                    x_t=x_t, s_t=s_t, T_s=Inf, α=0.0,
                    T_x=cfg.T, pair_uniform=false
                )
            end

            # ---- flatten edges ----
            edge_list = [(x...,) for ee in edge_list_tau for x in ee]
            reverse!(edge_list)

            windows = collect(1:length(edge_list))
            p_vals, smax, chi, _ = run_single_trial(N, edge_list, windows)

            idx = argmax(chi)
            push!(pc_vals, p_vals[idx])

            # ---- reconstruct graph at pc ----
            g_tmp = copy(g)
            for e in edge_list[1:idx]
                rem_edge!(g_tmp, e...)
            end

            # ---- sample paths ----
            adj2 = build_indexed_adjacency(g_tmp)
            st2 = SPPState_disorder(
                Vector{Int}(undef, nv(g_tmp)),
                Vector{Int}(undef, nv(g_tmp)),
                [Int[] for _ in 1:nv(g_tmp)],
                Vector{Int}(undef, nv(g_tmp)+1),
                adj2,
                falses(ne(g_tmp)),
                Int[],
                zeros(ne(g_tmp)),
                zeros(ne(g_tmp)),
                zeros(nv(g_tmp)),
            )

            paths_eids = Vector{Vector{Int}}()
            for _ in 1:sample_paths
                eids = nothing

                if cfg.mode == :pair_uniform
                    OD_pairs = find_node_pairs_within_distance(g_tmp, cfg.C)
                    o, d = sample(OD_pairs)
                    bfs_to!(o, d, st2, cfg.C)
                    result = sample_preferential_path_inf!(o, d, st2; T_x=cfg.T, α=0.0)
                    eids = result === nothing ? nothing : result[2]

                elseif cfg.mode == :source_uniform
                    o = rand(1:nv(g_tmp))
                    bfs_to_finite!(o, st2, cfg.C)
                    isempty(st2.seen) && continue
                    d = sample(st2.seen)
                    result = sample_preferential_path_inf!(o, d, st2; T_x=cfg.T, α=0.0)
                    eids = result === nothing ? nothing : result[2]

                else # mixed
                    o, d = sample(1:nv(g_tmp), 2, replace=false)
                    if !bfs_to!(o, d, st2, cfg.C)
                        isempty(st2.seen) && continue
                        d = sample(st2.seen)
                    end
                    result = sample_preferential_path_inf!(o, d, st2; T_x=cfg.T, α=0.0)
                    eids = result === nothing ? nothing : result[2]
                end

                eids !== nothing && push!(paths_eids, eids)
            end

            counts = count_edges(paths_eids, ne(g_tmp))
            push!(entropies, edge_entropy(counts) / log(ne(g_tmp)))
        end

        results[cfg] = (
            mean_entropy = mean(entropies),
            std_entropy  = std(entropies),
            mean_pc      = mean(pc_vals),
            std_pc       = std(pc_vals)
        )
    end

    return results
end

# ---------- Run ----------
res = run_entropy_experiment(;N=10000,num_trials=10)

println("\n=== Results ===")
for (k,v) in res
    println(k, " => ", v)
end

using Plots
# plot pc vs entropy for different configurations
plt = plot();
for (cfg, stats) in res
    scatter!(plt, [1 - stats.mean_pc], [stats.mean_entropy]; yerr=[stats.std_entropy], label="$(cfg.mode), C=$(cfg.C), T=$(cfg.T)")
end
xlabel!(plt, "Estimated pc")    
ylabel!(plt, "Normalized Edge Entropy at pc")