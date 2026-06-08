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
N_loop = 2 .^([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
rc_N = zeros(length(N_loop))
Sc_N = zeros(length(N_loop))
time = zeros(length(N_loop))
Tx = Inf
Ts = Inf
a = 0.0
beta=0.1
results_structs = []
for (i, N) in enumerate(N_loop)
    C = ceil(Int, N^(1/3))
    # C = 3      
    try
        d = load("Project1/sim_data/SW_C_N_1:3_cluster/Crossover_S/SW_PP_N$(N)_C$(C)_Tx$(Tx)_Ts$(Ts)_a$(a)_collated.jld2")
        res = d["result"]
        @info ("Loaded data for N=%d: Tx=%.2f, Ts=%.2f, a=%.1f", N, d["Tx"], d["Ts"], d["a"])
        print("N = $(N), num_trials = $(d["num_trials"])")
        push!(results_structs, res)
    catch e
        @warn "Failed to load data for N=$(N): $e"
        continue
    end
end
##
all_data = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}()
N_loop = [r.N for r in results_structs if r.N >= 10000]
rc_N = [r.pc for r in results_structs if r.N >= 10000]
rc_N_agg = [r.p_vals[argmax(r.chi)] for r in results_structs if r.N >= 10000]
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
# Estimate pc and nu using FSS
mean(rc_N)
mean(rc_N_agg)
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
y = log.(abs.(res.pc .- rc_N_agg))
mu = cov(x, y) / var(x)
# Calculate standard error of mu
residuals = y .- (mu .* x .+ mean(y .- mu .*x))
n = length(x)
std_mu = sqrt(sum(residuals.^2) / ((n-2) * sum((x .- mean(x)).^2)))

# fit
scatter(N_loop, abs.(rc_N_agg .- res.pc), xscale=:log10, yscale=:log10, marker=:circle, xlabel="N", ylabel="|pc(N) - pc|", label="Data")
plot!(N_loop, exp.(mu .* log.(N_loop) .+ mean(log.(abs.(rc_N_agg .- res.pc)) .- mu .* log.(N_loop))), label="Fit: |pc(N) - pc| ~ N^{-1/ν}, 1/ν=$(round(mu, digits=2))", lw=2)
ylims!(1e-4, 1e-1)

## 
νbar, intercept_nu, std_err_nu = Power_law_exponent(N_loop, std_rc_N)
plot(N_loop, std_rc_N, xscale=:log10, yscale=:log10, marker=:circle, xlabel="N", ylabel="std(p_c(N))", label="Data")
intercept = mean(log.((abs.(res.pc .- rc_N_agg))) .- νbar .* log.(N_loop))
plot(N_loop, exp.(νbar .* log.(N_loop) .+ intercept), label="Fit: std(p_c(N)) ~ N^{-1/ν}, 1/ν=$(round(νbar, digits=2))", lw=2)
scatter!(N_loop, abs.(rc_N_agg .- res.pc), xscale=:log10, yscale=:log10, marker=:circle, xlabel="N", ylabel="|pc(N) - pc|", label="Data")

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
scatter!(plt_pc, N_loop, std_rc_N, xscale=:log10, yscale=:log10, marker=:circle, xlabel="N", ylabel="σ(p_c(N))", label="")
plot!(plt_pc, N_loop, exp.(νbar .* log.(N_loop) .+ mean(log.(std_rc_N) .- νbar .* log.(N_loop))), label=label = L"1/\bar{\nu} = %$(round(νbar,digits=3)) \pm %$(round(std_err_nu,digits=3))", lw=2)
xlabel!(plt_pc, L"N")
ylabel!(plt_pc, L"σ(p_c(N))")
xlims!(1e4, maximum(N_loop)*10^0.4)
xticks!([1e4, 1e5, 1e6, 1e7])

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
xlims!(1e4, maximum(N_loop)*10^0.4)
xticks!([1e4, 1e5, 1e6, 1e7])

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
xlims!(1e4, maximum(N_loop)*10^0.4)
xticks!([1e4, 1e5, 1e6, 1e7])
# ---------------- Combine panels ----------------
final_plot = plot(
    plt_pc, plt_S, plt_chi,
    layout = (1,3),
    size = (1200, 360),   # 🔼 slightly larger figure
    margin = 6Plots.mm
)

display(final_plot)

savefig(final_plot, "Project1/Plot_paper/Estimates_Beta0.1_T$(Tx)_C_N_1_3_.png")


##
plot(N_loop,std_rc_N,xscale=:log10, yscale=:log10, marker=:circle, label="")
# estimate the slope from event-based ensemble
logN = log.(N_loop)
log_std = log.(std_rc_N)
mu = cov(logN, log_std) / var(logN)
plot!(N_loop, exp.(mu .* log.(N_loop) .+ mean(log_std .- mu .* log.(N_loop))), label="Std Dev of pc ~ N^{-$mu}", lw=2, linestyle=:dash)

# # find the pc 
# pc_candidates = range(mean(rc_N) - 0.01, mean(rc_N) + 0.01, length=1000)
# slope_curve = zeros(length(pc_candidates))
# r2 = zeros(length(pc_candidates))
# # find the best power law fit
# for (j, pc_cand) in enumerate(pc_candidates)
#     slope_curve[j] = cov(log.(N_loop), log.(abs.(rc_N .- pc_cand))) / var(log.(N_loop))
#     r2[j] = cor(log.(N_loop), log.(abs.(rc_N .- pc_cand)))^2
# end


# plot(pc_candidates, slope_curve, label="-β/ν", xlabel="p", ylabel="Slope of log-log fit", title="Slope of log-log fit vs p")
# plot(pc_candidates, r2, label="\$R^2\$", xlabel="p", ylabel="\$R^2(p)\$", yaxis=:right);
# idx = argmax(r2)
# res.pc 
# pc_candidates[idx]
# plot!([pc_candidates[idx]], [r2[idx]], seriestype=:scatter, label="Best fit", markershape=:diamond, markersize=8, color=:red)

# # closest negative slope to mu
# neg_idx = findall(<(0), slope_curve)
# neg_idx
# pc = isempty(neg_idx) ? pc_candidates[argmax(r2)] : pc_candidates[neg_idx[argmin(abs.(slope_curve[neg_idx] .- mu))]]
# scatter(N_loop,abs.(rc_N .- pc), xscale=:log10, yscale=:log10, marker=:circle, label="")
# #estimate the slope
# logN = log.(N_loop)
# log_diff = log.(abs.(rc_N .- pc))
# νbar = cov(logN, log_diff) / var(logN)
# plot!(N_loop, exp.(νbar .* log.(N_loop) .+ mean(log_diff .- νbar .* log.(N_loop))), label=L"|p_c(N) - p_c| ~ N^{-1/\bar{\nu}}, 1/\bar{\nu}=$(round(νbar, digits=2))", lw=2)

## data collapse conventional ensemble 
# color = palette(:viridis, length(results_structs))
# pc = res.pc
# plt_collapse_S = plot(
#     xlabel=L"(p - p_c) N^{1/ν}",
#     ylabel=L"S N^{β/ν}",
#     legend=false
# )

# for (N, (p_vals, S_vals, chi_vals)) in all_data
#     if N >= 10000
#         # 🔑 restrict to scaling window
#         mask = abs.(p_vals .- pc) .< 0.02

#         x = (p_vals[mask] .- pc) .* N^(-νbar)
#         y = S_vals[mask] .* N^(-βν)

#         plot!(plt_collapse_S, x, y, lw=2, alpha=0.7,color=color[findfirst(==(N), N_loop)])
#     end
# end
# display(plt_collapse_S)

# plt_collapse_chi = plot(
#     xlabel=L"(p - p_c) N^{1/ν}",
#     ylabel=L"χ N^{-γ/ν}",
#     legend=false
# )

# for (N, (p_vals, S_vals, chi_vals)) in all_data
#     if N >= 10000
#         mask = abs.(p_vals .- pc) .< 0.02

#         x = (p_vals[mask] .- pc) .* N^(-νbar)
#         y = chi_vals[mask] .* N^(-γν)

#         plot!(plt_collapse_chi, x, y, lw=2, alpha=0.7,color=color[findfirst(==(N), N_loop)] )
#     end
# end


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

display(plt_final_collapse)

savefig(plt_final_collapse, "Project1/Plot_paper/FSS_collapse_crossover_T$(Tx).png")

##
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

res = results_structs[end];
plot!(plt_path, res.path_average, ribbon=res.path_std, fillalpha=0.15, label="", color=:red,xlims=(0, 200000))
vline!(plt_path, [res.critical_trials_avg], color=:red, linestyle=:dash, alpha=0.5, label="")
xticks!(plt_path, 0:40000:200000)
plot!(plt_path,margin=6Plots.mm)

# for (i, r) in enumerate(results_structs)

#     if r.N < 10000
#         @info "Skipping path plot for N=$(r.N) due to large size"
#         continue
#     end

#     plot!(plt_path,

#         r.path_average,

#         ribbon=r.path_std,

#         fillalpha=0.15,

#         label="N=$(r.N)",

#         color=colors[i]

#     )

#     vline!(plt_path, [r.critical_trials_avg],

#         color=colors[i],

#         linestyle=:dash,

#         alpha=0.5,

#         label=false

#     )

# end
# xlims!(plt_path, (0, 80000))
# plot!(plt_path, legend=:topright,margin=6Plots.mm)



# =========================

# 3. Scaling of max path

# =========================
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

# savefig(plt_max, "Project1/Plot_paper/abstract2.png")
# =========================

# Final layout

# =========================

plot(plt_path, plt_max,

    layout=(1,2),

    size=(1000,400),

    margin=6Plots.mm

)


savefig("Project1/Plot_paper/SW_path_scaling_crossover_T$(Tx)_source_uni.png")

##
using Interpolations

# ─── Fitted exponents (from your right panel) ────────────────────────────────
α_peak     = μ
α_critical = μ_critical       # or use μ_critical from your fit

# ─── Per-N derived quantities ─────────────────────────────────────────────────
N_loop     = [r.N for r in results_structs]
t_peak_idx = [argmax(r.path_average)            for r in results_structs]
t_c_idx    = [round(Int, r.critical_trials_avg) for r in results_structs]
t_peak     = float.(t_peak_idx)
t_c        = float.(t_c_idx)

colors = palette(:viridis, length(results_structs));


# ══════════════════════════════════════════════════════════════════════════════
# CRITICAL COLLAPSE  with best ν
# ══════════════════════════════════════════════════════════════════════════════
p_t = [cumsum(r.path_average) ./ sum(r.path_average) for r in results_structs];
colors = palette(:viridis, length(results_structs));

plt = plot(

    xlabel = "t",

    ylabel = "p",

    legend = false

);
plt2 = plot(

    xlabel = "p",

    ylabel = L"l_b",

    legend = false,

);

slope = zeros(2,length(results_structs))
for (i, r) in enumerate(results_structs)

    y1 = p_t[i]

    x1 = 1:length(y1)

    x2 = p_t[i]
    y2 = r.path_average
    # choose at most 1000 evenly spaced indices

    idx1 = round.(Int, range(1, length(x1), length=min(40000, length(x1))))
    idx2 = round.(Int, range(1, length(x2), length=min(40000, length(x2))))

    plot!(

        plt,

        x1[idx1],

        y1[idx1],

        label = "N=$(r.N)",

        color=colors[i]

    )
    # find the derivative at the critical point 
    slope[1,i] = (diff(y1) ./ diff(x1))[t_c_idx[i]]   
    slope[2,i] = (diff(y1) ./ diff(x1))[t_peak_idx[i]]
    vline!(plt, [t_c[i]], color=colors[i], linestyle=:dash, label = "")

    plot!(

        plt2,

        x2[idx2],

        y2[idx2],

        label = "N=$(r.N)",

        color = colors[i]

    )
    vline!(plt2, [1 - r.pc], color=colors[i], linestyle=:dash, label = "")
    hline!(plt2, [r.N^(1/3)], color=colors[i], linestyle=:dot, label = "",alpha=0.7)

end

# plot(N_loop,slope,xscale=:log10, yscale=:log10, marker=:circle, label="")
# # estimate the slope 
# logN = log.(N_loop)
# log_slope = log.(slope)
# μ_slope = cov(logN, log_slope) / var(logN)
# plot!(N_loop, exp.(μ_slope .* logN .+ mean(log_slope .- μ_slope .* logN)), label="Slope ~ N^{$(round(μ_slope, digits=2))}", lw=2, linestyle=:dash)  

# xlims!(plt, (0, 40000))
display(plt)
display(plt2)

##
logN = log.(N_loop)
log_slope = log.(slope[1,:])
μ_slope = cov(logN, log_slope) / var(logN)
# peak_slope = cov(logN, log.(slope[2,:])) / var(logN)
logN = log.(N_loop)
log_slope = log.(slope[2,:])
peak_slope = cov(logN, log_slope) / var(logN)

plt_crit = plot(

    xlabel = L"(t - t_c)\, / N^{1 - α_{c} - 1/\bar{\nu}}",

    ylabel = L"l / N^{α_{c}}",

    legend = :topright

);

for (i, r) in enumerate(results_structs)
    r.N < 10_000 && continue
    T   = length(r.path_average)
    1-α_critical+νbar
    xs  = ((1:T) .- t_c[i])   ./ r.N^(1-α_critical+νbar)  # add correction to x-axis scaling to improve collapse
    ys  = r.path_average          ./ r.N^α_critical # optional log correction
    mask = -4 .< xs .< 4  # zoom into critical window
    plot!(plt_crit, xs[mask], ys[mask],
        ribbon    = (r.path_std ./ r.N^α_critical)[mask],
        fillalpha = 0.10,
        label     = "",
        color     = colors[i]
    )
end
ylims!(plt_crit, (-2, 6))
savefig(plt_crit, "Project1/Plot_paper/SW_path_collapse_critical_crossover_T$(Tx)_source_uni.png")
##
# plt_peak = plot(
#     title  = "Peak collapse",
#     legend = :topright
# );

# for (i, r) in enumerate(results_structs)
#     r.N < 10_000 && continue
#     T   = length(r.path_average)
#     xs  = (1:T)   .- t_peak[i] ./ r.N^(1-peak_slope)
#     ys  = r.path_average   ./ r.N^α_peak
#     mask = -100 .< xs .< 100      # zoom into critical window
#     plot!(plt_peak, xs[mask], ys[mask],
#         ribbon   = (r.path_std ./ r.N^α_peak)[mask],
#         fillalpha = 0.10,
#         label    = "N=$(r.N)",
#         color    = colors[i]
#     )
# end
# ylims!(plt_peak, (-2, 6))
# plt_peak
