

## load HighTcrossover data
results_structs_highT_crossover = []
N_loop = 2 .^([10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23])
rc_N = zeros(length(N_loop))
Sc_N = zeros(length(N_loop))
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
        push!(results_structs_highT_crossover, res)
    catch e
        @warn "Failed to load data for N=$(N): $e"
        continue
    end
end

## load LowTcrossover data
results_structs_lowT_crossover = []
rc_N = zeros(length(N_loop))    
Sc_N = zeros(length(N_loop))
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
        push!(results_structs_lowT_crossover, res)
    catch e
        @warn "Failed to load data for N=$(N): $e"
        continue
    end
end

## 

rc_N = zeros(length(N_loop))
Sc_N = zeros(length(N_loop))
results_structs_highT = []
time = zeros(length(N_loop))
Tx = Inf
Ts = Inf
a = 0.0
for (i, N) in enumerate(N_loop)
    C = 3
    try
        d = load("Project1/sim_data/SW_C_3_cluster/SW_0.1_chunk_results/Finite/SW_PP_N$(N)_C$(C)_Tx$(Tx)_Ts$(Ts)_a$(a)_collated.jld2")
        res = d["result"]
        num_trials = d["num_trials"]
        println("Loaded data for N=$N, C=$C, Tx=$Tx, Ts=$Ts, a=$a with $num_trials trials")
        push!(results_structs_highT, res)
    catch
        @error "Failed to load data for N=$N"
        continue
    end
end

## 
rc_N = zeros(length(N_loop))
Sc_N = zeros(length(N_loop))
results_structs_lowT = []
time = zeros(length(N_loop))
Tx = 0.1
Ts = Inf
a = 0.0
for (i, N) in enumerate(N_loop)
    C = 3
    try
        d = load("Project1/sim_data/SW_C_3_cluster/SW_0.1_chunk_results/Finite/SW_PP_N$(N)_C$(C)_Tx$(Tx)_Ts$(Ts)_a$(a)_collated.jld2")
        res = d["result"]
        num_trials = d["num_trials"]
        println("Loaded data for N=$N, C=$C, Tx=$Tx, Ts=$Ts, a=$a with $num_trials trials")
        push!(results_structs_lowT, res)
    catch
        @error "Failed to load data for N=$N"
        continue
    end
end

##
using Plots
using LaTeXStrings
using Printf
include("../../FSS_core.jl")

gr()

# ============================================================
# COLORS
# ============================================================

# C = 3  → RED family
c3_high = :red
c3_low  = :darkred

# C = N^(1/3) → BLUE family
cn_high = :dodgerblue
cn_low  = :navy

# yshift 
shift_c3_high = 1
shift_c3_low = 1
shift_cn_low = 1
shift_cn_high = 1

datasets = [
    (
        "C=3 High",
        results_structs_highT,
        c3_high,
        shift_c3_high
    ),

    (
        "C=3 Low",
        results_structs_lowT,
        c3_low,
        shift_c3_low
    ),

    (
        "C=N^{1/3} High",
        results_structs_highT_crossover,
        cn_high,
        shift_cn_high
    ),

    (
        "C=N^{1/3} Low",
        results_structs_lowT_crossover,
        cn_low,
        shift_cn_low
    )
]

# ============================================================
# CREATE PANELS
# ============================================================

plt_pc = plot(
    xscale = :log10,
    yscale = :log10,
    xlabel = L"N",
    ylabel = L"|p_c(N)-p_c|",
    legend = :bottomleft
)

plt_S = plot(
    xscale = :log10,
    yscale = :log10,
    xlabel = L"N",
    ylabel = L"S_c(N)",
    legend = :bottomleft
)

plt_chi = plot(
    xscale = :log10,
    yscale = :log10,
    xlabel = L"N",
    ylabel = L"\chi_c(N)",
    legend = :bottomright
)

# ============================================================
# LOOP OVER DATASETS
# ============================================================

for (label, results_structs, color, shift) in datasets

    # ------------------------------------------
    # extract observables
    # ------------------------------------------

    rs = [r for r in results_structs if r.N >= 10000]

    N_loop = [r.N for r in rs]

    rc_N = [r.pc for r in rs]

    Sc_N = [r.Sc for r in rs]

    chi_N = [r.Xc for r in rs]

    std_Sc_N = [r.std_Sc for r in rs]

    std_chi_N = [r.std_Xc for r in rs]

    for res in results_structs
        if res.N >= 10000
            all_data[res.N] = (res.p_vals, res.GCC_frac, res.chi)
        end
    end
    βν, intercept, std_err = Power_law_exponent(N_loop, Sc_N)
    res = estimate_pc_slope(all_data; rc_N=mean(rc_N), delta=0.04, ngrid=100, target_slope=βν)
    pc_est = res.pc

    # ------------------------------------------
    # fits
    # ------------------------------------------

    βν, intercept, std_err_s = Power_law_exponent(N_loop, Sc_N)
    γν, intercept_chi, std_err_chi = Power_law_exponent(N_loop, chi_N)
    νbar, intercept_nu, std_err_nu = Power_law_exponent(N_loop, abs.(rc_N .- res.pc))



    # ========================================================
    # PANEL 1 : pc scaling
    # ========================================================

    scatter!(
        plt_pc,
        N_loop, abs.(rc_N .- res.pc),
        marker = :circle,
        color = color,
        label = "$(label)"
    )

    plot!(
        plt_pc,
        N_loop,
        exp.(νbar .* log.(N_loop) .+ intercept_nu),
        color = color,
        lw = 2,
        label = "1/ν=$(round(νbar,digits=2))"
    )

    # ========================================================
    # PANEL 2 : Sc scaling
    # ========================================================

    scatter!(
        plt_S,
        N_loop,
        Sc_N,
        marker = :circle,
        color = color,
        label = "$(label)"
    )

    plot!(
        plt_S,
        N_loop,
        exp.(βν .* log.(N_loop) .+ intercept),
        color = color,
        lw = 2,
        label = "β/ν=$(round(βν,digits=2))"
    )

    # ========================================================
    # PANEL 3 : chi scaling
    # ========================================================

    scatter!(
        plt_chi,
        N_loop,
        chi_N,
        marker = :circle,
        color = color,
        label = "$(label)"
    )

    plot!(
        plt_chi,
        N_loop,
        exp.(γν .* log.(N_loop) .+ intercept_chi),
        color = color,
        lw = 2,
        label = "γ/ν=$(round(γν,digits=2))"
    )

end

# ============================================================
# FINAL LAYOUT
# ============================================================

final_plot = plot(
    plt_pc,
    plt_S,
    plt_chi,
    layout = (1,3),
    size = (1400, 420),
    margin = 6Plots.mm
)

display(final_plot)

# savefig(
#     final_plot,
#     "Project1/Plot_paper/Combined_critical_exponents_comparison.png"
# )
