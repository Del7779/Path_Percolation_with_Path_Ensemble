##
using JLD2
using Graphs 
using Plots
using StatsBase
using LsqFit
using Interpolations
using LaTeXStrings
using Printf
include("../FSS_core.jl")

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

##
N_loop = 2 .^([10, 11, 12, 13, 14, 15, 16])
rc_N = zeros(length(N_loop))
Sc_N = zeros(length(N_loop))
results_structs_pair = []
time = zeros(length(N_loop))
Tx = Inf
Ts = Inf
a = 0.0
beta=0.1
for (i, N) in enumerate(N_loop)
    C = ceil(Int, N^(1/3))
    # C = 3      
    try
        d = load("Project1/sim_data/SW_C_N_1:3_cluster/Pair-uniform/SW_PP_pair_uniform_N$(N)_C$(C)_Tx$(Tx)_Ts$(Ts)_a$(a)_collated.jld2")
        res = d["result"]
        print("N = $(N), num_trials = $(d["num_trials"])")
        push!(results_structs_pair, res)
    catch e
        @warn "Failed to load data for N=$(N): $e"
        continue
    end
end

plot(results_structs_pair[end].p_vals, results_structs_pair[end].GCC_frac, label="pair-uniform high T", xlabel="p", ylabel="S")


## 
# N_loop = 2 .^([10, 11, 12, 13, 14, 15,16,17, 18, 19, 20, 21, 22, 23])
rc_N = zeros(length(N_loop))
Sc_N = zeros(length(N_loop))
results_structs_source_lowT = []
time = zeros(length(N_loop))
Tx = 0.2
Ts = Inf
a = 0.0
beta=0.1
for (i, N) in enumerate(N_loop)
    C = ceil(Int, N^(1/3))
    # C = 3      
    try
        d = load("Project1/sim_data/SW_C_N_1:3_cluster/Crossover_S/SW_PP_N$(N)_C$(C)_Tx$(Tx)_Ts$(Ts)_a$(a)_collated.jld2")
        res = d["result"]
        @info ("Loaded data for N=%d: Tx=%.2f, Ts=%.2f, a=%.1f", N, d["Tx"], d["Ts"], d["a"])
        print("N = $(N), num_trials = $(d["num_trials"])")
        push!(results_structs_source_lowT, res)
    catch e
        @warn "Failed to load data for N=$(N): $e"
        continue
    end
end

##
# N_loop = 2 .^([10, 11, 12, 13, 14, 15,16,17, 18, 19, 20, 21, 22, 23])
rc_N = zeros(length(N_loop))
Sc_N = zeros(length(N_loop))
results_structs_source_highT = []
time = zeros(length(N_loop))
Tx = Inf
Ts = Inf
a = 0.0
beta=0.1
for (i, N) in enumerate(N_loop)
    C = ceil(Int, N^(1/3))
    # C = 3      
    try
        d = load("Project1/sim_data/SW_C_N_1:3_cluster/Crossover_S/SW_PP_N$(N)_C$(C)_Tx$(Tx)_Ts$(Ts)_a$(a)_collated.jld2")
        res = d["result"]
        @info ("Loaded data for N=%d: Tx=%.2f, Ts=%.2f, a=%.1f", N, d["Tx"], d["Ts"], d["a"])
        print("N = $(N), num_trials = $(d["num_trials"])")
        push!(results_structs_source_highT, res)
    catch e
        @warn "Failed to load data for N=$(N): $e"
        continue
    end
end

plt=plot();
for (i, res) in enumerate(results_structs_source_lowT)
    windows = round.(Int, range(1, stop=length(res.p_vals), length=1000))
    vline!(plt, [res.pc], label="source-uniform low T, N=$(N_loop[i])", xlabel="p", ylabel="S", linestyle=:dash)
    plot!(plt,results_structs_source_lowT[i].p_vals[windows], results_structs_source_lowT[i].GCC_frac[windows], label="source-uniform low T, N=$(N_loop[i])", xlabel="p", ylabel="S")
    
end
plt

##
pltt = plot();
res_source_highT = results_structs_source_highT[end];
res_source_lowT = results_structs_source_lowT[end];
res_pair = results_structs_pair[end];
windows = round.(Int, range(1, stop=length(res_source_highT.p_vals), length=1000))
plot!(pltt, 1 .- res_source_highT.p_vals[windows], res_source_highT.GCC_frac[windows], color=:red, label="source-uniform high T", xlabel="p", ylabel="S")
windows2 = round.(Int, range(1, stop=length(res_source_lowT.p_vals), length=1000))
plot!(pltt, 1 .- res_source_lowT.p_vals[windows2], res_source_lowT.GCC_frac[windows2], color=:blue, label="source-uniform low T", xlabel="p", ylabel="S")
plot!(pltt, 1 .- res_pair.p_vals, res_pair.GCC_frac, color=:green, label="pair-uniform high T", xlabel="p", ylabel="S")
pltt
