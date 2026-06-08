##
using JLD2
using Graphs 
using Plots
using StatsBase
using LsqFit
using Interpolations
using LaTeXStrings
using Printf
include("../../FSS_core.jl")
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
plt = plot();
N_loop = 2 .^([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
C_loop = round.(Int, 3 .* ones(length(N_loop)))

# low T
rc_N = zeros(length(N_loop))
Sc_N = zeros(length(N_loop))
results_structs_lowT = []
time = zeros(length(N_loop))
Tx = 0.1
Ts = Inf
a = 0.0
for (i, N) in enumerate(N_loop)
    C = C_loop[i]
    try
        d = load("Project1/sim_data/SW_C_3_cluster/SW_0.1_chunk_results/Finite_3/SW_PP_N$(N)_C$(C)_Tx$(Tx)_Ts$(Ts)_a$(a)_collated.jld2")
        res = d["result"]
        num_trials = d["num_trials"]
        println("Loaded data for N=$N, C=$C, Tx=$Tx, Ts=$Ts, a=$a with $num_trials trials")
        push!(results_structs_lowT, res)
    catch e
        @error "Failed to load data for N=$N: $e"
        continue
    end
end
##
N_loop = 2 .^([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
C_loop = round.(Int, 3 .* ones(length(N_loop)))

# low T
rc_N = zeros(length(N_loop))
Sc_N = zeros(length(N_loop))
results_structs_highT = []
time = zeros(length(N_loop))
Tx = Inf
Ts = Inf
a = 0.0
for (i, N) in enumerate(N_loop)
    C = C_loop[i]
    try
        d = load("Project1/sim_data/SW_C_3_cluster/SW_0.1_chunk_results/Finite_3/SW_PP_N$(N)_C$(C)_Tx$(Tx)_Ts$(Ts)_a$(a)_collated.jld2")
        res = d["result"]
        num_trials = d["num_trials"]
        println("Loaded data for N=$N, C=$C, Tx=$Tx, Ts=$Ts, a=$a with $num_trials trials")
        push!(results_structs_highT, res)
    catch e
        @error "Failed to load data for N=$N: $e"
        
        continue
    end
end

## Pair uniform
@load "/Users/del/Julia/Project1/sim_data/SW_C_3_cluster/old_cluster_results/SW_SPP_C3_B0.1_results_partial.jld2" results_structs
res_pair_uniform_lowT = results_structs[end]; 

@load "/Users/del/Julia/Project1/sim_data/SW_C_3_cluster/old_cluster_results/SW_PP_C3_B0.1_results_partial.jld2" results_structs
res_pair_uniform_highT = results_structs[end];

##
plt = plot();
res_highT = results_structs_highT[end];
res_lowT = results_structs_lowT[end];
num_points_plot = 1000
window = round.(Int, range(1, stop=length(res_highT.p_vals), length=num_points_plot))
plot!(plt, 1 .- res_highT.p_vals[window], res_highT.GCC_frac[window], xlabel=L"p", ylabel=L"S", label=L"T = ∞", legend=:topright,color=:red)
window2 = round.(Int, range(1, stop=length(res_lowT.p_vals), length=num_points_plot))
plot!(plt, 1 .- res_lowT.p_vals[window2], res_lowT.GCC_frac[window2], xlabel=L"p", ylabel=L"S", label=L"T = 0.1", legend=:topright,color=:blue)
windows3 = round.(Int, range(1, stop=length(res_pair_uniform_highT.p_vals), length=num_points_plot))
plot!(plt, 1 .- res_pair_uniform_highT.p_vals[windows3], res_pair_uniform_highT.GCC_frac[windows3], label="",alpha=0.2,color=:red)
windows4 = round.(Int, range(1, stop=length(res_pair_uniform_lowT.p_vals), length=num_points_plot))
plot!(plt, 1 .- res_pair_uniform_lowT.p_vals[windows4], res_pair_uniform_lowT.GCC_frac[windows4], label="",alpha=0.2,color=:blue)   
savefig(plt, "Project1/Plot_paper/SW_phase_diagram_source_uniform.png")

##
results_structs = results_structs_lowT;
all_data = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}()
N_loop = [r.N for r in results_structs if r.N >= 10000]
# rc_N = [r.p_vals[argmax(r.chi)] for r in results_structs if r.N >= 10000]
rc_N = [r.pc for r in results_structs if r.N >= 10000]
Sc_N = [r.Sc for r in results_structs if r.N >= 10000]
n_N = [r.N for r in results_structs if r.N >= 10000]
chi_N = [r.Xc for r in results_structs if r.N >= 10000]
std_Sc_N = [r.std_Sc for r in results_structs if r.N >= 10000]
std_chi_N = [r.std_Xc for r in results_structs if r.N >= 10000]
std_rc_N = [r.std_pc for r in results_structs if r.N >= 10000]
for res in results_structs
    if res.N >= 10000
        all_data[res.N] = (res.p_vals, res.GCC_frac, res.chi)
    end
end

## calculate β/ν from Sc scaling
βν, intercept, std_err = Power_law_exponent(N_loop, Sc_N)
plot(N_loop, Sc_N, xscale=:log10, yerror = std_Sc_N, yscale=:log10, marker=:circle, xlabel="N", ylabel="χ",label="");
plot!(N_loop, exp.(βν .* log.(N_loop) .+ intercept), label="\$χ ∼ N^{γ/ν}\$, γ/ν=$(round(βν, digits=3)) ± $(round(std_err, digits=3))", lw=2)
println("Estimated β/ν = $βν ± $std_err")
# Estimate pc and nu using FSS
res = estimate_pc_slope(all_data; rc_N=mean(rc_N), delta=0.04, ngrid=100, target_slope=βν)

# p1 = res.pc + diff(res.p_candidates)[1]
# p2 = res.pc - diff(res.p_candidates)[1]
# p_loop = [res.pc]
# plt = plot(legend=:bottomleft);
# for p in p_loop
#     N, s, chi = get_vals_at_p(p,all_data)
#     scatter!(plt, N, s, xscale=:log10, yscale=:log10, marker=:circle, xlabel="N", ylabel="S", label="$p")
#     slope, intercept, std_err = Power_law_exponent(N, s)
#     plot!(plt, N, exp.(slope .* log.(N) .+ intercept), label="Fit at p=$(round(p, digits=3)), slope=$(round(slope, digits=3)) ± $(round(std_err, digits=3))", lw=2)
# end

## Diagnostic plot R2 and Slope
plot(res.p_candidates, res.slope_curve, label="-β/ν", xlabel="p", ylabel="β/ν", title="Slope of log-log fit vs p")
plot!(res.p_candidates, res.r2, label="\$R^2\$", xlabel="p", ylabel="\$R^2(p)\$", yaxis=:right)
plot!([res.pc], [res.slope], seriestype=:scatter, label="", markershape=:diamond, markersize=8, color=:red)
plot!([res.pc], [res.r2[findfirst(==(res.pc), res.p_candidates)]], seriestype=:scatter, label="", markershape=:star, markersize=8, color=:green, yaxis=:right)
vline!([mean(rc_N)], seriestype=:scatter, label="\$<p_{c}(N)>\$", markershape=:circle, markersize=8, color=:blue,legend=false,linestyle=:dash)
hline!([βν], label="Target β/ν = $(round(βν, digits=3))", linestyle=:dash, color=:black)
plot!(legend=:bottomright)

# ---------------- Format exponents ----------------
##
using Plots
using Printf
using LaTeXStrings

gr()
# ---------------- Format exponents ----------------
βν, intercept, std_err_s = Power_law_exponent(N_loop, Sc_N)
γν, intercept_chi, std_err_chi = Power_law_exponent(N_loop, chi_N)
νbar, intercept_nu, std_err_nu = Power_law_exponent(N_loop, abs.(rc_N .- res.pc))

βν_str = @sprintf("%.3f ± %.3f", βν, std_err_s)
γν_str = @sprintf("%.3f ± %.3f", γν, std_err_chi)
νbar_str = @sprintf("%.3f ± %.3f", νbar, std_err_nu)


# ---------------- (a) pc scaling ----------------
plt_pc = plot(legend = :bottomleft);
scatter!(plt_pc, N_loop, abs.(rc_N .- res.pc), xscale=:log10, yscale=:log10, marker=:circle, xlabel="N", ylabel="|pc(N) - pc|", label="")
plot!(plt_pc, N_loop, exp.(νbar .* log.(N_loop) .+ mean(log.(abs.(rc_N .- res.pc)) .- νbar .* log.(N_loop))), label=label = L"1/\bar{\nu} = %$(round(νbar,digits=3)) \pm %$(round(std_err_nu,digits=3))", lw=2)
xlabel!(plt_pc, L"N")
ylabel!(plt_pc, L"|p_c(N)-p_c|")
# ylims!(10^(-2.5), 10^(-1.5))

# ---------------- (b) Sc scaling ----------------
plt_S = plot(legend = :bottomleft);
scatter!(plt_S,
    N_loop,
    Sc_N,
    yerror = std_Sc_N,
    xscale = :log10,
    yscale = :log10,
    label = ""
)
plot!(plt_S,
    N_loop, exp.(βν .* log.(N_loop) .+ intercept), label=L"\beta/\nu= %$(round(-βν, digits=3))\pm %$(round(std_err_s, digits=3))", lw=2)
xlabel!(plt_S, L"N")
ylabel!(plt_S, L"S_c(N)")
xlims!(1e4, maximum(N_loop)*10^0.1)

# ---------------- (c) chi scaling ----------------
plt_chi = plot(legend = :bottomright)
fit_chi = exp.(γν .* log.(N_loop) .+ intercept_chi)

scatter!(plt_chi,
    N_loop,
    chi_N,
    yerror = std_chi_N,
    xscale = :log10,
    yscale = :log10,
    label = ""
)

plot!(plt_chi,
    N_loop, fit_chi, label = L"\gamma/\nu= %$(round(γν, digits=3)) \pm %$(round(std_err_chi, digits=3))", lw=2) # 🔥 using the fitted line for the label to show the exact exponent and error
xlabel!(plt_chi, L"N")
ylabel!(plt_chi, L"\chi_c(N)")
xlims!(1e4, maximum(N_loop)*10^0.1)

# ---------------- Combine panels ----------------
final_plot = plot(
    plt_pc, plt_S, plt_chi,
    layout = (1,3),
    size = (1200, 360),   # 🔼 slightly larger figure
    margin = 6Plots.mm
)

display(final_plot)

savefig(final_plot, "Project1/Plot_paper/Estimate_C_3_source_lowT.png")

##
using StatsBase
using KernelDensity
function density_curve!(plt, x; label="", color=nothing, lw=2.2, alpha=0.9)

    x = collect(skipmissing(x))

    x = x[isfinite.(x)]

    if length(x) < 5

        @warn "Too few points for KDE: $label"

        return plt

    end

    kd = kde(x)

    plot!(

        plt,

        kd.x,

        kd.density,

        label = label,

        linewidth = lw,

        alpha = alpha,

        color = color

    )

    return plt

end

# ------------------------------------------------------------


colors = palette(:viridis, length(results_structs))

# ------------------------------------------------------------

# 1. Pseudo-critical point distribution collapse

# ------------------------------------------------------------

# plt_pc = plot(

#     xlabel = L"(p_c^{(i)} - p_c)N^{1/\bar{\nu}}",

#     ylabel = "Density",

#     legend = :topright,

# );

# for (i, res) in enumerate(results_structs)

#     x_pc = (res.seudo_pc .- pc) .* res.N^(-νbar)

#     density_curve!(

#         plt_pc,

#         x_pc;

#         label = L"N=%$(res.N)",

#         color = colors[i]

#     )

# end
# plt_pc

# ------------------------------------------------------------

# 2. Order-parameter distribution collapse

# ------------------------------------------------------------
plt_Sc = plot(

    xlabel = L"S_c N^{\beta/\bar{\nu}}",

    ylabel = "Density",

    legend = :topright,

);

for (i, res) in enumerate(results_structs)

    if res.N <= 10000
        continue
    end
    x_Sc = res.seudo_Sc .* res.N^(-βν)

    density_curve!(

        plt_Sc,

        x_Sc;

        label = L"N=%$(res.N)",

        color = colors[i]

    )

end

# ------------------------------------------------------------

# 3. Susceptibility distribution collapse

# ------------------------------------------------------------

plt_chi = plot(

    xlabel = L"\chi_c N^{-\gamma/\bar{\nu}}",

    ylabel = "Density",

    legend = :topright,

);

for (i, res) in enumerate(results_structs)

    if res.N <= 10000
        continue
    end

    x_chi = res.seudo_chi .* res.N^(-γν)

    density_curve!(

        plt_chi,

        x_chi;

        label = L"N=%$(res.N)",

        color = colors[i]

    )

end

# ------------------------------------------------------------

# Combine figure

# ------------------------------------------------------------

plt_all = plot(

    plt_Sc,

    plt_chi,

    layout = (1, 2),

    size = (1200, 420),

    margin = 6Plots.mm,

)

plt_final_collapse = plot( 

    plt_Sc,

    plt_chi,

    layout = (1, 2),

    size = (1200, 400),

    margin = 6Plots.mm,

)

display(plt_final_collapse)

savefig(plt_final_collapse, "Project1/Plot_paper/FSS_collapse_finite_lowT.png")




