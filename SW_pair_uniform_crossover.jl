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
N_loop = 2 .^([10, 11, 12, 13, 14, 15, 16, 17, 18])
rc_N = zeros(length(N_loop))
Sc_N = zeros(length(N_loop))
results_structs = []
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
        push!(results_structs, res)
    catch e
        @warn "Failed to load data for N=$(N): $e"
        continue
    end
end
##
colors = palette(:viridis, length(results_structs));

# =========================

# 1. Raw path evolution

# =========================

plt_path = plot(

    xlabel=L"t",

    ylabel=L"l",

    legend=:topleft

);

for (i, r) in enumerate(results_structs)

    # if r.N < 10000
    #     @info "Skipping path plot for N=$(r.N) due to large size"
    #     continue
    # end

    plot!(plt_path,

        r.path_average,

        ribbon=r.path_std,

        fillalpha=0.15,

        label="N=$(r.N)",

        color=colors[i]

    )

    vline!(plt_path, [r.critical_trials_avg],

        color=colors[i],

        linestyle=:dash,

        alpha=0.5,

        label=false

    )

end
xlims!(plt_path, (0, 10000))
plot!(plt_path, legend=:topright,margin=6Plots.mm)

# =========================

# # 2. Scaling collapse

# # =========================

# plt_collapse = plot(

#     xlabel=L"(t - t_c)/N^{2/3}",

#     ylabel=L"\ell / N^{0.283}",

#     legend=false

# );

# for (i, r) in enumerate(results_structs)

#     t = 1:length(r.path_average)

#     tc = argmax(r.path_average)

#     mask = (t .>= tc - 20) .& (t .<= tc + 20)

#     x = 4*r.N ./ (t[mask] .- tc)

#     y = r.path_average[mask]

#     plot!(plt_collapse,

#         x, y,

#         color=colors[i],

#         alpha=0.7

#     )

# end

# plt_collapse

# =========================

# 3. Scaling of max path

# =========================

##
plt_max = plot(

    xscale=:log10,

    yscale=:log10,

    xlabel=L"N",

    ylabel=L"l",

    legend=:bottomright

);

N_loop = [r.N for r in results_structs]

max_path = [maximum(r.path_average) for r in results_structs]

critical_path = [r.path_average[round(Int, r.critical_trials_avg)] for r in results_structs]

max_path -  ceil.(Int, N_loop .^ (1/3))

scatter!(plt_max, N_loop, max_path,

    marker=:circle,

    label="\$l_{max}\$"

);

scatter!(plt_max, N_loop, critical_path,

    marker=:diamond,

    label="\$l_{c}\$"

);

# reference scaling

plot!(plt_max, N_loop, N_loop .^ (1/3),

    linestyle=:dash,

    label="\$N^{1/3}\$"

);

# fit max scaling

logN = log.(N_loop)

μ = cov(logN, log.(max_path)) / var(logN)

plot!(plt_max,

    N_loop,

    exp.(μ .* logN .+ mean(log.(max_path) .- μ .* logN)),

    label="\$N^{$(round(μ, digits=2))}\$",

    linewidth=2

)

# fit critical path scaling
μ_critical = cov(logN, log.(critical_path)) / var(logN)
plot!(plt_max,

    N_loop,

    exp.(μ_critical .* logN .+ mean(log.(critical_path) .- μ_critical .* logN)),

    label="\$N^{$(round(μ_critical, digits=2))}\$",

    linewidth=2

)
plot!(plt_max,legend=:topleft)

## 

# savefig(plt_max, "Project1/Plot_paper/abstract2.png")
# =========================

# Final layout

# =========================

plot(plt_path, plt_max,

    layout=(1,2),

    size=(1000,400),

    margin=6Plots.mm

)


# savefig("Project1/Plot_paper/SW_path_scaling_crossover_T$(Tx)_source_uni.png")
#----------------------

##
all_data = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}()
N_loop = [r.N for r in results_structs]
rc_N = [r.pc for r in results_structs]
rc_N_agg = [r.p_vals[argmax(r.chi)] for r in results_structs]
Sc_N = [r.Sc for r in results_structs]
n_N = [r.N for r in results_structs]
chi_N = [r.Xc for r in results_structs]
std_Sc_N = [r.std_Sc for r in results_structs]
std_chi_N = [r.std_Xc for r in results_structs]
std_rc_N = [r.std_pc for r in results_structs ]
for res in results_structs
    if res.N >= 10000
        all_data[res.N] = (res.p_vals, res.GCC_frac, res.chi)
    end
end

## calculate β/ν from Sc scaling
βν, intercept, std_err = Power_law_exponent(N_loop, Sc_N)
plot(N_loop, Sc_N, xscale=:log10, yerror = std_Sc_N, yscale=:log10, xlabel="N", ylabel=L"S_c",label="");
plot!(N_loop, exp.(βν .* log.(N_loop) .+ intercept), label="\$S_c ∼ N^{β/ν}\$, β/ν=$(round(βν, digits=2))", lw=2)
println("Estimated β/ν = $βν ± $std_err")
# Estimate pc and nu using FSS
mean(rc_N)
res = estimate_pc_slope(all_data; rc_N=mean(rc_N), delta=0.04, ngrid=100, target_slope=βν)

## Diagnostic plot R2 and Slope
plot(res.p_candidates, res.slope_curve, label="-β/ν", xlabel="p", ylabel="β/ν", title="Slope of log-log fit vs p")
plot!(res.p_candidates, res.r2, label="\$R^2\$", xlabel="p", ylabel="\$R^2(p)\$", yaxis=:right)
plot!([res.pc], [res.slope], seriestype=:scatter, label="", markershape=:diamond, markersize=8, color=:red)
plot!([res.pc], [res.r2[findfirst(==(res.pc), res.p_candidates)]], seriestype=:scatter, label="", markershape=:star, markersize=8, color=:green, yaxis=:right)
vline!([mean(rc_N)], seriestype=:scatter, label="\$<p_{c}(N)>\$", markershape=:circle, markersize=8, color=:blue,legend=false,linestyle=:dash)
hline!([βν], label="Target β/ν = $(round(βν, digits=2))", linestyle=:dash, color=:black)
plot!(legend=:bottomright)



# calculate 1/ν from the scaling of |pc(N) - pc| vs N
x = log.(N_loop)
y = log.(std_rc_N)
mu = cov(x, y) / var(x)
# Calculate standard error of mu
residuals = y .- (mu .* x .+ mean(y .- mu .*x))
n = length(x)
std_mu = sqrt(sum(residuals.^2) / ((n-2) * sum((x .- mean(x)).^2)))


##

using Plots
using Printf
using LaTeXStrings

gr()
# ---------------- Format exponents ----------------
βν, intercept, std_err_s = Power_law_exponent(N_loop, Sc_N)
γν, intercept_chi, std_err_chi = Power_law_exponent(N_loop, chi_N)
νbar, intercept_nu, std_err_nu = Power_law_exponent(N_loop, std_rc_N)
βν_str = @sprintf("%.3f ± %.3f", βν, std_err_s)
γν_str = @sprintf("%.3f ± %.3f", γν, std_err_chi)
νbar_str = @sprintf("%.3f ± %.3f", νbar, std_err_nu)


# ---------------- (a) pc scaling ----------------
plt_pc = plot(legend = :bottomleft);
scatter!(plt_pc, N_loop, abs.(rc_N_agg .- res.pc), xscale=:log10, yscale=:log10, marker=:circle, xlabel="N", ylabel="|pc(N) - pc|", label="")
plot!(plt_pc, N_loop, exp.(νbar .* log.(N_loop) .+ mean(log.(abs.(rc_N_agg .- res.pc)) .- νbar .* log.(N_loop))), label="1/ν=$(round(νbar, digits=2)) ± $(round(std_err_nu, digits=3))", lw=2)
xlabel!(plt_pc, L"N")
ylabel!(plt_pc, L"|p_c(N)-p_c|")
xlims!(1e3, maximum(N_loop)*10^0.1)

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
    N_loop, exp.(βν .* log.(N_loop) .+ intercept), label="β/ν=$(round(βν, digits=3)) ± $(round(std_err_s, digits=2))", lw=2)
xlabel!(plt_S, L"N")
ylabel!(plt_S, L"S_c(N)")
xlims!(1e3, maximum(N_loop)*10^0.1)

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
    N_loop, fit_chi, label = "γ/ν=$(round(γν, digits=2)) ± $(round(std_err_chi, digits=3))", lw=2) # 🔥 using the fitted line for the label to show the exact exponent and error
xlabel!(plt_chi, L"N")
ylabel!(plt_chi, L"\chi_c(N)")
xlims!(1e3, maximum(N_loop)*10^0.1)

# ---------------- Combine panels ----------------
final_plot = plot(
    plt_pc, plt_S, plt_chi,
    layout = (1,3),
    size = (1200, 360),   # 🔼 slightly larger figure
    margin = 6Plots.mm
)

final_plot
# savefig(final_plot, "Project1/Plot_paper/Estimates_Beta0.1_T$(Tx)_C_N_1_3_.png")


## data collapse conventional ensemble 
color = palette(:viridis, length(results_structs))
pc = res.pc
plt_collapse_S = plot(
    xlabel=L"(p - p_c) N^{1/ν}",
    ylabel=L"S N^{β/ν}",
    legend=false
)

for (N, (p_vals, S_vals, chi_vals)) in all_data
    if N >= 10000
        # 🔑 restrict to scaling window
        mask = abs.(p_vals .- pc) .< 0.02

        x = (p_vals[mask] .- pc) .* N^(-νbar)
        y = S_vals[mask] .* N^(-βν)

        plot!(plt_collapse_S, x, y, lw=2, alpha=0.7,color=color[findfirst(==(N), N_loop)])
    end
end
display(plt_collapse_S)

plt_collapse_chi = plot(
    xlabel=L"(p - p_c) N^{1/ν}",
    ylabel=L"χ N^{-γ/ν}",
    legend=false
)

for (N, (p_vals, S_vals, chi_vals)) in all_data
    if N >= 10000
        mask = abs.(p_vals .- pc) .< 0.02

        x = (p_vals[mask] .- pc) .* N^(-νbar)
        y = chi_vals[mask] .* N^(-γν)

        plot!(plt_collapse_chi, x, y, lw=2, alpha=0.7,color=color[findfirst(==(N), N_loop)] )
    end
end


## probability collapse Sc(N) and Pc(N)
# ------------------------------------------------------------
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

plt_final_collapse
# savefig(plt_final_collapse, "Project1/Plot_paper/FSS_collapse_crossover_T$(Tx).png")