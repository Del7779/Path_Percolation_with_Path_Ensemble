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
# struct SPPResults
#     N                   :: Int
#     p_vals              :: Vector{Float64}
#     GCC_frac            :: Vector{Float64}
#     pc                  :: Float64
#     Sc                  :: Float64
#     Xc                  :: Float64
#     seudo_pc            :: Vector{Float64}
#     seudo_Sc            :: Vector{Float64}
#     seudo_chi           :: Vector{Float64}
#     std_pc              :: Float64
#     std_Sc              :: Float64
#     std_Xc              :: Float64
#     chi                 :: Vector{Float64}
#     epi                 :: Vector{Float64}
#     path_average        :: Vector{Float64}
#     path_std            :: Vector{Float64}
#     deg_width_average   :: Vector{Float64}
#     critical_trials_avg :: Float64
#     critical_trials_std :: Float64
# end
##
beta_loop1 = [0.1,0.2,0.3,0.4,0.5,0.8,1]
beta_loop2 = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99]
beta_loop = sort(vcat(beta_loop1, beta_loop2))
C = 3
model = ["SPP","PP"]

spp_rc = []
rpp_rc = []
spp_p = []
rpp_p = []
spp_mu = []
rpp_mu = []
        
for (j,model) in enumerate(model)
    for (i,beta) in enumerate(beta_loop)
        all_data = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}()
        println("Loading data for model = $model, C = $C, β = $beta")
        @load "/Users/del/Julia/Project1/sim_data/SW_C_3_cluster/old_cluster_results/SW_$(model)_C3_B$(beta_loop[i])_results_partial.jld2" results_structs
          
        N_loop = [r.N for r in results_structs if r.N >= 10000]
        rc_N = 1 .- [r.pc for r in results_structs if r.N >= 10000]
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
        x = log.(N_loop)
        y = log.(Sc_N)
        # Calculate slope using covariance and variance
        βν = cov(x, y) / var(x)
        println("Estimated β/ν = $βν for β = $beta")
        @info "$(mean(rc_N)) is the mean of rc_N for model = $model, β = $beta"
        res = estimate_pc_slope(all_data; rc_N=mean(rc_N), delta=0.03, ngrid=100, target_slope=βν)
        @info "Estimated pc using FSS for model = $model, β = $beta: $(res.pc) with β/ν = $(res.slope)"
        x = log.(N_loop)
        y = log.(abs.(res.pc .- rc_N))
        mu = cov(x, y) / var(x)
        ## std of mu
        residuals = y .- (mu .* x .+ mean(y .- mu .* x))
        n = length(x)
        std_mu = sqrt(sum(residuals.^2) / ((n-2) * sum((x .- mean(x)).^2)))

        @info "Estimated 1/ν  = $mu for model = $model, β = $beta"
        if j == 1
            push!(spp_rc, res.pc)
                push!(spp_p, mu)
                push!(spp_mu, std_mu)   
        else
            push!(rpp_rc, res.pc)
                push!(rpp_p, mu)
                push!(rpp_mu, std_mu)   
        end
    end
end


##

# --- Colors ---
c1 = :blue
c2 = :red

# --- Panel 1: Δpc vs β ---
plt1 = scatter(
    beta_loop,
    abs.(spp_rc .- rpp_rc),
    marker=:circle,
    color=:black,
    markerstrokecolor=:black,
    xlabel="β",
    ylabel="Δ\$p_c\$",
    label="",
)

# --- Panel 2: pc vs β ---
plt2 = plot()

plot!(
    plt2,
    beta_loop,
    1 .- spp_rc,
    yerror=0.001,
    marker=:circle,
    color=c1,
    label="T = 0.1",
)

plot!(
    plt2,
    beta_loop,
    1 .- rpp_rc,
    yerror=0.001,
    marker=:square,
    color=c2,
    label="T = ∞",
)

xlabel!(plt2, "β")
ylabel!(plt2, "\$p_c\$")

# --- Combine panels ---
plt = plot(
    plt2,
    plt1,
    layout=(1,2),
    dpi=300,
    size=(1200,500),
    margin=5Plots.mm
)
plt

# savefig(plt, "Project1/Plot_paper/Estimated_pc_vs_beta.png")

plt_exp = plot();

scatter!(
    plt_exp,
    beta_loop,
    spp_p,
    marker=:circle,
    color=c1,
    label="\$T_x\$ = 0.1", ylims=(-0.5, -0.2)
)

scatter!(
    plt_exp,
    beta_loop,
    rpp_p,
    marker=:square,
    color=c2,
    label="\$T_x\$ = ∞")

xlabel!(plt_exp, "β")
ylabel!(plt_exp, "\$1/ν\$")
ylims!(-0.7, 0)
hline!(plt_exp, [-1/3], linestyle=:dash, color=:black, label="\$1/ν\$ = 1/3")


# savefig(plt_exp, "Project1/Plot_paper/Exponent_vs_beta.png")


#---------------------------
## Phase diagram
beta_loop1 = [0.1,0.2,0.3,0.4,0.5,0.8,1]
beta_loop2 = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99]
beta_loop = sort(vcat(beta_loop1, beta_loop2))
C = 3
model = ["SPP","PP"]

using Plots
gr()
# ---------- Global style ----------
default(
    fontfamily="Computer Modern",
    linewidth=2,
    markersize=6,
    markerstrokewidth=1.5,
    grid=false,
    framestyle=:axis,
    legendfontsize=8,
    guidefontsize=12,
    tickfontsize=10,
    dpi=300
)

# shortest path percolation (SPP) results
@load "/Users/del/Julia/Project1/sim_data/SW_C_3_cluster/old_cluster_results/SW_$(model[1])_C3_B$(beta_loop[1])_results_partial.jld2" results_structs
res = results_structs[end]; 
epi = res.epi
res.N
p_vals = res.p_vals
GCC_frac = res.GCC_frac 
Chi = res.chi
color=:blue
num_points_plot = 1000
window = round.(Int, range(1, stop=length(p_vals), length=num_points_plot))
plt
plt
plot!(plt, 1 .- p_vals[window], GCC_frac[window], label="", legend=:bottomleft,framestyle=:axis,color=color,alpha=0.2);
# plt12 = twinx(plt1)
# plot!(plt12, 1 .- p_vals[window], Chi[window], ylabel="χ", label="", linestyle=:dash, color=color,yaxis = :right,framestyle=:axis)
plot!(plt1,framestyle=:box)

# Noise percolation (PP) results
@load "/Users/del/Julia/Project1/sim_data/SW_C_3_cluster/old_cluster_results/SW_$(model[2])_C3_B$(beta_loop[1])_results_partial.jld2" results_structs
res = results_structs[end];
p_vals = res.p_vals
GCC_frac = res.GCC_frac
Chi = res.chi
color = :red
window = round.(Int, range(1, stop=length(p_vals), length=num_points_plot))
plot!(plt, 1 .- p_vals[window], GCC_frac[window], label="", legend=:topright,color=color,alpha=0.2);
plot!(plt12, 1 .- p_vals[window], Chi[window], ylabel="χ",label="",color=color,linestyle=:dash)
plot!(plt1,framestyle=:axis)

# savefig(plt1, "Project1/Plot_paper/abstract.png")

#-----------------------------------
## Universality
using Plots
using Printf
using LaTeXStrings
beta_loop1 = [0.1,0.2,0.3,0.4,0.5,0.8,1]
beta_loop2 = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99]
beta_loop = sort(vcat(beta_loop1, beta_loop2))
C = 3
model = ["SPP","PP"]
##
i = 1
j = 1
@load "Project1/sim_data/SW_C_3_cluster/old_cluster_results/SW_$(model[i])_C3_B$(beta_loop[j])_results_partial.jld2" results_structs
res = results_structs[end];
res.N
window = round.(Int, range(1, stop=length(res.p_vals), length=1000))
plot(res.p_vals[window], res.GCC_frac[window], xlabel="p", ylabel="S", label="pair-uniform", legend=:bottomleft)
vline!(1 .- [res.pc], label="\$p_c\$", linestyle=:dash, color=:black)
## second batch 
N_loop = 2 .^([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
C_loop = round.(Int, 3 .* ones(length(N_loop)))
rc_N = zeros(length(N_loop))
Sc_N = zeros(length(N_loop))
results_structs = []
time = zeros(length(N_loop))
Tx = Inf
Ts = Inf
a = 0.0
for (i, N) in enumerate(N_loop)
    C = C_loop[i]
    try
        d = load("Project1/sim_data/SW_C_3_cluster/SW_0.1_chunk_results/SW_PP_N$(N)_C$(C)_Tx$(Tx)_Ts$(Ts)_a$(a)_collated.jld2")
        res = d["result"]
        num_trials = d["num_trials"]
        println("Loaded data for N=$N, C=$C, Tx=$Tx, Ts=$Ts, a=$a with $num_trials trials")
        push!(results_structs, res)
    catch
        @error "Failed to load data for N=$N"
        continue
    end
end
##

all_data = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}()
N_loop = [r.N for r in results_structs if r.N >= 10000]
# rc_N = [r.p_vals[argmax(r.chi)] for r in results_structs if r.N >= 10000]
rc_N = [1 - r.pc for r in results_structs if r.N >= 10000]
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
# βν, intercept, std_err_s = Power_law_exponent(N_loop, Sc_N)
βν = res.slope
intercept = mean(log.(Sc_N) .- βν .* log.(N_loop))
γν, intercept_chi, std_err_chi = Power_law_exponent(N_loop, chi_N)
νbar, intercept_nu, std_err_nu = Power_law_exponent(N_loop, abs.(rc_N .- res.pc))
# estimate \nubar from std_rc_N and N 
plot(N_loop, std_rc_N, xscale=:log10, yscale=:log10, marker=:circle, xlabel="N", ylabel="std(p_c(N))", label="")
#estimate slope
νbar, intercept_nu, std_err_nu = Power_law_exponent(N_loop, std_rc_N)

βν_str = @sprintf("%.3f ± %.3f", βν, std_err_s)
γν_str = @sprintf("%.3f ± %.3f", γν, std_err_chi)
νbar_str = @sprintf("%.3f ± %.3f", νbar, std_err_nu)

##
# ---------------- (a) pc scaling ----------------
plt_pc = plot(legend = :bottomleft);
std_rc_N
scatter!(plt_pc, N_loop, abs.(rc_N .- res.pc), xscale=:log10, yscale=:log10, xlabel=L"N", ylabel=L"|p_c(N) - p_c|", label="")
plot!(plt_pc, N_loop, exp.(νbar .* log.(N_loop) .+ mean(log.(abs.(rc_N .- res.pc)) .- νbar .* log.(N_loop))), label="1/ν=$(round(νbar, digits=3)) ± $(round(std_err_nu, digits=3))", lw=2)
xlabel!(plt_pc, L"N")
ylabel!(plt_pc, L"|p_c(N)-p_c|")

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
    N_loop, exp.(βν .* log.(N_loop) .+ intercept), label="β/ν=$(round(βν, digits=3)) ± $(round(std_err_s, digits=3))", lw=2)
xlabel!(plt_S, L"N")
ylabel!(plt_S, L"S_c(N)")

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
    N_loop, fit_chi, label = "\$γ/ν=$(round(γν, digits=3)) ± $(round(std_err_chi, digits=3))\$", lw=2) # 🔥 using the fitted line for the label to show the exact exponent and error
xlabel!(plt_chi, L"N")
ylabel!(plt_chi, L"\chi_c(N)")

# ---------------- Combine panels ----------------
final_plot = plot(
    plt_pc, plt_S, plt_chi,
    layout = (1,3),
    size = (1200, 360),   # 🔼 slightly larger figure
    margin = 6Plots.mm
)
final_plot
savefig(final_plot, "Project1/Plot_paper/Estimate_criticalexp_$(model[i])_B$(beta_loop[j]).png")

