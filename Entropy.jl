## ============================================================
##  Path-Percolation Entropy Simulation
##  Usage: julia entropy_simulation.jl <N> <beta>
## ============================================================

# using Distributed
# using SlurmClusterManager

# # ── CLI arguments ────────────────────────────────────────────
# const NUM_NODES = parse(Int,   ARGS[1])
# const BETA      = parse(Float64, ARGS[2])
# const HORIZON         = parse(Int,   ARGS[3])
# # ── Output directory ─────────────────────────────────────────
# const DIR = "/data/math-path-percol/orie5068/Path_Based_Percolation_chunk_run/entropy_data"
# isdir(DIR) || mkpath(DIR)

# # ── Spin up workers ──────────────────────────────────────────
# addprocs(SlurmManager())
# @info "Workers online: $(nworkers())"

# # ============================================================
# #  Distributed definitions
# # ============================================================
# @everywhere begin
#     using Graphs
#     using Random, Distributions
#     using StatsBase
#     using ProgressMeter
#     using BenchmarkTools
#     using SlurmClusterManager

#     include("Newman_Ziff.jl")
#     include("user_core_weighted.jl")
#     include("user_core_infinite_prefwalk.jl")

#     # ── Edge / node usage counts ─────────────────────────────
#     """
#     Accumulate algorithmic betweenness counts for edges and nodes
#     from a collection of sampled paths.
#     """
#     function count_edges_nodes(
#         path_eids::Vector{Vector{Int}},
#         paths::Vector{Vector{Int}},
#         num_edges::Int,
#         num_nodes::Int,
#     )
#         counts_edges = zeros(Float64, num_edges)
#         counts_nodes = zeros(Float64, num_nodes)

#         for eids in path_eids
#             @inbounds for eid in eids
#                 counts_edges[eid] += 1.0
#             end
#         end
#         for path in paths
#             @inbounds for node in path
#                 counts_nodes[node] += 1.0
#             end
#         end

#         return counts_edges, counts_nodes
#     end

#     # ── Participation ratio ──────────────────────────────────
#     """
#     Inverse participation ratio of a usage distribution.
#     Pass `num_packets` to normalise by total flow rather than sum.
#     """
#     function participation_ratio(counts; num_packets=nothing)
#         p = num_packets === nothing ? counts ./ sum(counts) : counts ./ num_packets
#         return 1.0 / sum(p .^ 2)
#     end

#     # ── Shannon entropy of edge usage ────────────────────────
#     """
#     Shannon entropy of the edge-usage distribution,
#     optionally normalised by `num_packets`.
#     """
#     function edge_entropy(counts_edges; num_packets=nothing)
#         p = num_packets === nothing ? counts_edges ./ sum(counts_edges) : counts_edges ./ num_packets
#         return -sum(p .* log.(p .+ eps()))
#     end
# end  # @everywhere

# ##
# # ============================================================
# #  Graph construction
# # ============================================================
# const g            = newman_watts_strogatz(NUM_NODES, 4, BETA)
# const num_nodes    = nv(g)
# const num_edges    = ne(g)
# const base_degrees = degree(g)

# const ROUTING_TEMPS    = [0.1, 0.6, 1.5, Inf]
# const NUM_TRIALS       = 50
# const BOND_PERCOLATION = false   # false → site percolation order

# # Storage
# pc_tx          = zeros(length(ROUTING_TEMPS))
# save_edge_list_full = Dict{Float64, Vector{Vector{Tuple{Int,Int}}}}()
# chi_peak_idx   = Dict{Float64, Vector{Int}}()


# # ============================================================
# #  Main loop — one temperature at a time
# # ============================================================
# for (temp_idx, T_x) in enumerate(ROUTING_TEMPS)

#     T_s      = Inf
#     od_pairs = find_node_pairs_within_distance(g, HORIZON)
#     adj      = build_indexed_adjacency(g)

#     @info "T_x = $T_x | OD pairs: $(length(od_pairs)) | $(Base.summarysize(od_pairs)/1e6) MB"

#     # Broadcast read-only data to workers once per temperature
#     @everywhere od_pairs_ref = $od_pairs
#     @everywhere adj_ref      = $adj

#     x_weights = ones(num_edges)
#     s_weights = ones(num_edges)

#     # ── Parallel trials ──────────────────────────────────────
#     results = @showprogress pmap(1:NUM_TRIALS) do _
#         st = SPPState_disorder(
#             Vector{Int}(undef, num_nodes),
#             Vector{Int}(undef, num_nodes),
#             [Int[] for _ in 1:num_nodes],
#             Vector{Int}(undef, num_nodes + 1),
#             adj_ref,
#             falses(num_edges),
#             Vector{Int64}(),
#             Vector{Float64}(undef, num_edges),
#             Vector{Float64}(undef, num_edges),
#             Vector{Float64}(undef, num_nodes),
#         )
#         init_sppstate_disorder!(st, base_degrees, x_weights, s_weights)

#         _, _, edge_list_tau, _, _ = run_single_pp_disorder(
#             num_nodes, HORIZON, od_pairs_ref, st;
#             T_x=T_x, T_s=T_s, α=0.0,
#         )

#         # Reverse: percolation adds edges from empty → full
#         edge_list = reverse!([(x...,) for ee in edge_list_tau for x in ee])

#         windows = collect(1:num_edges)
#         p_vals, s_max, chi, _, _ = run_single_trial(
#             num_nodes, edge_list, windows; shuffle=BOND_PERCOLATION
#         )

#         # Pseudo-critical point: steepest jump in giant component
#         crit_idx   = argmax(diff(s_max)) + 1
#         pseudo_pc  = p_vals[crit_idx]

#         # Critical trial (time-step at which the critical edge was removed)
#         critical_edge  = edge_list[crit_idx]
#         critical_trial = findfirst(tau -> critical_edge in tau, edge_list_tau)

#         # Return only what is needed downstream
#         (p_vals, s_max, chi, edge_list[:], critical_edge, critical_trial, pseudo_pc)
#     end

#     # ── Aggregate results ────────────────────────────────────
#     num_points = length(results[1][1])
#     s_max_sum  = zeros(num_points)
#     chi_sum    = zeros(num_points)
#     pseudo_pcs = zeros(NUM_TRIALS)

#     for (j, res) in enumerate(results)
#         (_, s_max_j, chi_j, tail_edges, _, _, pseudo_pc_j) = res

#         pseudo_pcs[j] = pseudo_pc_j
#         s_max_sum    .+= s_max_j
#         chi_sum      .+= chi_j

#         push!(get!(chi_peak_idx,   T_x, Int[]),              argmax(chi_j))
#         push!(get!(save_edge_list_full, T_x, Vector{Vector{Tuple{Int,Int}}}()), tail_edges)
#     end

#     pc_tx[temp_idx] = mean(pseudo_pcs)

#     s_max_avg = s_max_sum ./ NUM_TRIALS
#     chi_avg   = chi_sum   ./ NUM_TRIALS
#     gcc_frac  = s_max_avg ./ num_nodes   # giant-component fraction

#     @info "T_x = $T_x | p_c ≈ $(round(pc_tx[temp_idx], digits=4))"
# end


# # ============================================================
# #  Snapshot entropy / participation-ratio profiles
# # ============================================================
# const NUM_SAMPLE_SNAPSHOTS = 20
# const NUM_SAMPLE_TRIALS    = 15

# # ── Fixed evaluation grid (shared across all trials and temperatures) ──


# PR_t     = zeros(length(ROUTING_TEMPS), NUM_SAMPLE_SNAPSHOTS)
# PR_t_std = zeros(length(ROUTING_TEMPS), NUM_SAMPLE_SNAPSHOTS)
# S_t      = zeros(length(ROUTING_TEMPS), NUM_SAMPLE_SNAPSHOTS)
# S_t_std  = zeros(length(ROUTING_TEMPS), NUM_SAMPLE_SNAPSHOTS)

# for (temp_idx, T_x) in enumerate(ROUTING_TEMPS)
#     p_grid = range(0.0, stop=(1 - pc_tx[temp_idx]), length=NUM_SAMPLE_SNAPSHOTS) |> collect
#     PR_mat = zeros(NUM_SAMPLE_SNAPSHOTS, NUM_SAMPLE_TRIALS)
#     S_mat  = zeros(NUM_SAMPLE_SNAPSHOTS, NUM_SAMPLE_TRIALS)

#     trial_sample_ids = rand(1:NUM_TRIALS, NUM_SAMPLE_TRIALS)

#     for (j, trial_idx) in enumerate(trial_sample_ids)

#         # Full removal sequence for this trial (saved reversed in main loop)
#         full_removal_seq = save_edge_list_full[T_x][trial_idx]

#         for (k, p_target) in enumerate(p_grid)
#             # Exact number of edges to remove to reach this p
#             n_remove = round(Int, p_target * num_edges)
#             n_remove = clamp(n_remove, 0, length(full_removal_seq))

#             g_snap = copy(g)
#             for edge in full_removal_seq[1:n_remove]
#                 rem_edge!(g_snap, edge...)
#             end

#             snap_edges   = ne(g_snap)
#             snap_nodes   = nv(g_snap)
#             snap_degrees = degree(g_snap)
#             snap_adj     = build_indexed_adjacency(g_snap)
#             snap_od      = find_node_pairs_within_distance(g_snap, HORIZON)

#             st_snap = SPPState_disorder(
#                 Vector{Int}(undef, snap_nodes),
#                 Vector{Int}(undef, snap_nodes),
#                 [Int[] for _ in 1:snap_nodes],
#                 Vector{Int}(undef, snap_nodes + 1),
#                 snap_adj,
#                 falses(snap_edges),
#                 Vector{Int64}(),
#                 Vector{Float64}(undef, snap_edges),
#                 Vector{Float64}(undef, snap_edges),
#                 Vector{Float64}(undef, snap_nodes),
#             )
#             init_sppstate_disorder!(st_snap, snap_degrees, ones(snap_edges), ones(snap_edges))

#             num_path_samples = round(Int, mean(snap_degrees) * snap_nodes)
#             paths_bag     = Vector{Vector{Int}}(undef, 0)
#             path_eids_bag = Vector{Vector{Int}}(undef, 0)

#             for _ in 1:num_path_samples
#                 (o, d) = sample(snap_od)
#                 bfs_to!(o, d, st_snap, HORIZON)
#                 path, path_eids = sample_preferential_path_inf!(
#                     o, d, st_snap;
#                     T_x=T_x, α=0.0,
#                     use_dist_constraint=true, max_steps=0,
#                 )
#                 push!(paths_bag,     path)
#                 push!(path_eids_bag, path_eids)
#             end

#             counts_edges, _ = count_edges_nodes(
#                 path_eids_bag, paths_bag, snap_edges, snap_nodes
#             )

#             PR_mat[k, j] = participation_ratio(counts_edges) / snap_edges
#             S_mat[k, j]  = edge_entropy(counts_edges) / log(snap_edges)
#         end
#     end

#     @info "T_x = $T_x | snapshot scan complete"

#     PR_t[temp_idx, :]     = vec(mean(PR_mat, dims=2))
#     PR_t_std[temp_idx, :] = vec(std(PR_mat,  dims=2))
#     S_t[temp_idx, :]      = vec(mean(S_mat,  dims=2))
#     S_t_std[temp_idx, :]  = vec(std(S_mat,   dims=2))
# end


##plot
using Plots
using JLD2
using LaTeXStrings
using Printf

gr()
# ---------- Global style ----------
default(
    fontfamily="Computer Modern",
    linewidth=2,
    markersize=8,
    markerstrokewidth=1.5,
    grid=false,
    framestyle=:box,
    legendfontsize=8,
    guidefontsize=12,
    tickfontsize=10,
    dpi=300
)
beta = 0.1
@load "/Users/del/Julia/Project1/sim_data/Entropy_data/entropy_results_N1000000_beta$(beta).jld2"
# @load "/Users/del/Julia/Project1/sim_data/Entropy_data/entropy_results_N1000000_beta1.0.jld2"
ROUTING_TEMPS    = [0.1, 0.6, 1.5, Inf]
using Plots, Printf
plt_beta_0 = plot(legend=false);
plt3 = plot(legend=false);
C = 3
for (i,Tx) in enumerate(ROUTING_TEMPS)
    p_grid = range(0.0, stop=(1 - pc_tx[i]), length=NUM_SAMPLE_SNAPSHOTS) |> collect
    plot!(plt_beta_0, p_grid, S_t[i, :], yerror=(S_t_std[i, :]), xlabel=L"p", ylabel=L"H", marker=:circle, label="\$T=$(Tx)\$, C=$(C)\$, beta=$(beta)\$", color=RGB(i / length(ROUTING_TEMPS), 0, 1 - i / length(ROUTING_TEMPS)))
    plot!(plt3, p_grid, PR_t[i, :], yerror=(PR_t_std[i, :]), xlabel=L"p", ylabel=L"PR", marker=:diamond, label="\$T=$(Tx)\$, C=$(C)\$, beta=$(beta)\$", color=RGB(i / length(ROUTING_TEMPS), 0, 1 - i / length(ROUTING_TEMPS)))
    vline!(plt_beta_0, 1 .- [pc_tx[i]], label="", color=RGB(i / length(ROUTING_TEMPS), 0, 1 - i / length(ROUTING_TEMPS)), linestyle=:dash)
    vline!(plt3, 1 .- [pc_tx[i]], label="", color=RGB(i / length(ROUTING_TEMPS), 0, 1 - i / length(ROUTING_TEMPS)), linestyle=:dash)
end
pp1 = plot(plt_beta_0,plt3, layout=(1, 2), size=(1400, 600), margin=12Plots.mm,dpi=600);
pp1
plt_beta_0
savefig(plt_beta_0,"Project1/Plot_paper/entropy_S_t_beta$(beta).png")
savefig(pp1,"Project1/Plot_paper/entropy_PR_t_beta$(beta).png")

## beta = 1 
beta = 1
@load "/Users/del/Julia/Project1/sim_data/Entropy_data/entropy_results_N1000000_beta1.0.jld2"
ROUTING_TEMPS    = [0.1, 0.6, 1.5, Inf]
using Plots, Printf
plt_beta1 = plot(legend=false);
plt3 = plot(legend=false);
C = 3
for (i,Tx) in enumerate(ROUTING_TEMPS)
    p_grid = range(0.0, stop=(1 - pc_tx[i]), length=NUM_SAMPLE_SNAPSHOTS) |> collect
    plot!(plt_beta1, p_grid, S_t[i, :], yerror=(S_t_std[i, :]), xlabel=L"p", ylabel=L"H", marker=:circle, label="\$T=$(Tx)\$, C=$(C)\$, beta=$(beta)\$", color=RGB(i / length(ROUTING_TEMPS), 0, 1 - i / length(ROUTING_TEMPS)))
    plot!(plt3, p_grid, PR_t[i, :], yerror=(PR_t_std[i, :]), xlabel=L"p", ylabel=L"PR", marker=:diamond, label="\$T=$(Tx)\$, C=$(C)\$, beta=$(beta)\$", color=RGB(i / length(ROUTING_TEMPS), 0, 1 - i / length(ROUTING_TEMPS)))
    vline!(plt_beta1, 1 .- [pc_tx[i]], label="", color=RGB(i / length(ROUTING_TEMPS), 0, 1 - i / length(ROUTING_TEMPS)), linestyle=:dash)
    vline!(plt3, 1 .- [pc_tx[i]], label="", color=RGB(i / length(ROUTING_TEMPS), 0, 1 - i / length(ROUTING_TEMPS)), linestyle=:dash)
end
pp1 = plot(plt_beta1,plt3, layout=(1, 2), size=(1400, 600), margin=12Plots.mm,dpi=600);
pp1
plt_beta1
savefig(plt_beta1,"Project1/Plot_paper/entropy_S_t_beta$(beta).png")

pp_entropy_2 = plot(plt_beta_0, plt_beta1, layout=(1, 2), size=(1400, 600), margin=6Plots.mm,dpi=600);
pp_entropy_2
savefig(pp_entropy_2,"Project1/Plot_paper/entropy_H_beta_comparison.png")