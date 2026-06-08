using Graphs, JLD2, Distributed, Statistics, ProgressMeter,Distributed
using Plots
using Printf
using LaTeXStrings
using Plots
using LsqFit, StatsBase
using JLD2, Statistics,Interpolations

include("../Newman_Ziff.jl")

N_loop = 2 .^(15:22)
num_trials = 1000
beta = 1
BATCH_SIZE = 20  # 🔥 important

save_dir = "Project1/sim_data/test_local/ER_bond_percolation"
isdir(save_dir) || mkpath(save_dir)
##
for (i,N) in enumerate(N_loop)

    # g = newman_watts_strogatz(N, 4, beta)
    g = erdos_renyi(N, Int(4*N/2))   # for testing only
    edge_list = [(src(e), dst(e)) for e in edges(g)]
    windows = collect(1:ne(g))

    m = length(windows)

    # =====================
    # Accumulators
    # =====================
    s_max_acc = zeros(Float64, m)
    chi_acc = zeros(Float64, m)
    epsilon_acc = zeros(Float64, m)
    k_acc = zeros(Float64, m)

    peudo_pc_acc = zeros(Float64, num_trials)
    peudo_s_acc = zeros(Float64, num_trials)
    peudo_chi_acc = zeros(Float64, num_trials)

    p_vals = nothing

    # =====================
    # Batched pmap
    # =====================
    @showprogress for batch_start in 1:BATCH_SIZE:num_trials

        batch_end = min(batch_start + BATCH_SIZE - 1, num_trials)
        batch_range = batch_start:batch_end

        trial_results = pmap(batch_range) do trial
            p, s, chi, epsilon, K = run_single_trial(nv(g), edge_list, windows; shuffle=true)
            idx = argmax(chi)
            return (p, s, chi, epsilon, K,
                    p[idx],
                    s[idx],
                    chi[idx])
        end
        # =====================
        # Accumulate immediately
        # =====================
        for (local_i, res) in enumerate(trial_results)

            global_i = batch_start + local_i - 1

            p, s, chi, epsilon, K,
            peudo_pc, peudo_s, peudo_chi = res

            if p_vals === nothing
                p_vals = p   # store once
            end

            s_max_acc .+= s
            chi_acc .+= chi
            epsilon_acc .+= epsilon
            k_acc .+= K

            peudo_pc_acc[global_i] = peudo_pc
            peudo_s_acc[global_i] = peudo_s
            peudo_chi_acc[global_i] = peudo_chi
        end
        ## memory
        @info "Memory usage after batch: $(Base.summarysize(trial_results) / 1e6) MB"
        # =====================
        # Free memory EARLY
        # =====================
        trial_results = nothing
        GC.gc(true)
        @info "Completed trials $(batch_end / num_trials * 100)% for N=$(N)"
    end

    # =====================
    # Final averages
    # =====================
    inv_t = 1.0 / num_trials

    s_max_acc .*= inv_t
    chi_acc .*= inv_t
    epsilon_acc .*= inv_t
    k_acc .*= inv_t

    pc_n = mean(peudo_pc_acc)
    sc_n = mean(peudo_s_acc)
    chi_n = mean(peudo_chi_acc)
    srd_pc = std(peudo_pc_acc)
    srd_s = std(peudo_s_acc)
    srd_chi = std(peudo_chi_acc)

    result = (
        p = p_vals,
        s_max = s_max_acc,
        chi = chi_acc,
        epsilon = epsilon_acc,
        K = k_acc,
        peudo_pc = pc_n,
        peudo_s = sc_n,
        peudo_chi = chi_n,
        srd_pc = srd_pc,
        srd_s = srd_s,
        srd_chi = srd_chi
    )

    savepath = joinpath(save_dir, "ER_bond_percolation_N$(N).jld2")
    @save savepath result

    @info "Saved results for N=$(N)"
end

##
using Plots
plt = plot();
plt2 =plot();
save_dir = "Project1/sim_data/test_local/SW_bond_percolation"

for N in N_loop
    savepath = joinpath(save_dir, "SW_bond_percolation_N$(N).jld2")
    @load savepath result
    windows = round.(Int, range(1, length(result.p), 100))
    plot!(plt2,result.p[windows], result.epsilon[windows], label="N=$(N)", xlabel="p", ylabel="excessive degree per node", title="Model=ER Bond Percolation, β=$(0.1)")
    plot!(plt, result.p[windows], result.s_max[windows] ./ N, label="N=$(N)", xlabel="p", ylabel="s_max/N", title="Bond Percolation on ER Network")
    vline!(plt2, [result.peudo_pc], label="pc(N=$(N))", linestyle=:dash)
end

plt
plt2

##
## load
N_loop = 2 .^(15:22)
rc_N = zeros(Float64, length(N_loop))
rc_N2 = zeros(Float64, length(N_loop))
Sc_N = zeros(Float64, length(N_loop))
chi_N = zeros(Float64, length(N_loop))
std_Sc_N = zeros(Float64, length(N_loop))
std_chi_N = zeros(Float64, length(N_loop))  
std_Rc_N = zeros(Float64, length(N_loop))
save_dir = "Project1/sim_data/test_local/SW_bond_percolation"
for N in N_loop
    try 
    savepath = joinpath(save_dir, "SW_bond_percolation_N$(N).jld2")
    @load savepath result
    rc_N2[findfirst(==(N), N_loop)] = result.p[argmax(result.chi)]

    rc_N[findfirst(==(N), N_loop)] = result.peudo_pc
    Sc_N[findfirst(==(N), N_loop)] = result.peudo_s./N
    chi_N[findfirst(==(N), N_loop)] = result.peudo_chi
    std_Rc_N[findfirst(==(N), N_loop)] = result.srd_pc
    std_Sc_N[findfirst(==(N), N_loop)] = result.srd_s./N^2
    std_chi_N[findfirst(==(N), N_loop)] = result.srd_chi
    catch e
        @warn "Failed to load results for N=$(N): $e"
    end
end
all_data = Dict{Int, Tuple}()
for N in N_loop
    savepath = joinpath(save_dir, "SW_bond_percolation_N$(N).jld2")
    @load savepath result

    P_vals = result.s_max ./ N
    S_vals = result.chi

    all_data[N] = (result.p, P_vals, S_vals)
end

using Interpolations

function get_vals_at_p(p_target, all_data)
    N_vals = Float64[]
    P_vals = Float64[]
    S_vals = Float64[]

    for (N, (p, P, S)) in all_data
        itpP = LinearInterpolation(p, P)
        itpS = LinearInterpolation(p, S)

        push!(N_vals, N)
        push!(P_vals, itpP(p_target))
        push!(S_vals, itpS(p_target))
    end

    return N_vals, P_vals, S_vals
end

## estimtae in the even-based ensemble 
x = log.(N_loop)
y = log.(Sc_N)
# Calculate slope using covariance and variance
βν = cov(x, y) / var(x)
# Calculate standard error of slope
residuals = y .- (βν .* x .+ mean(y .- (βν).*x))
n = length(x)
std_err = sqrt(sum(residuals.^2) / ((n-2) * sum((x .- mean(x)).^2)))
println("β/ν = $(βν) ± $std_err")
x = log.(N_loop)
y_chi = log.(chi_N)
γν = cov(x, y_chi) / var(x)
# Calculate standard error of slope for χc
residuals_chi = y_chi .- (γν .* x .+ mean(y_chi .- (γν).*x))
std_err_chi = sqrt(sum(residuals_chi.^2) / ((n-2) * sum((x .- mean(x)).^2)))
println("γ/ν = $(round(γν,digits=3)) ± $(round(std_err_chi,digits=3))")


## 
pc_guess = mean(rc_N)
delta = 0.02
p_candidates = range(pc_guess - delta, pc_guess + delta, length=50)

residual_loop = zeros(length(p_candidates))
r2_loop = fill(NaN, length(p_candidates))
slope_loop = fill(NaN, length(p_candidates))

best_p = nothing
best_score = Inf   # minimize distance to expected slope
best_slope = nothing

target_slope =  βν  # expected slope from scaling theory

for (i, p_try) in enumerate(p_candidates)

    N_vals, P_vals, _ = get_vals_at_p(p_try, all_data)

    idx = sortperm(N_vals)
    N_sub = N_vals[idx]
    P_sub = P_vals[idx]

    # remove tiny values (important!)
    mask = P_sub .> 1e-12
    if sum(mask) < 3
        continue
    end

    logN = log.(N_sub[mask])
    logP = log.(P_sub[mask])

    slope = cov(logN, logP) / var(logN)

    fit = slope .* logN .+ mean(logP .- slope .* logN)

    residual = sum((logP .- fit).^2)

    y_mean = mean(logP)
    ss_tot = sum((logP .- y_mean).^2)

    if ss_tot < 1e-12
        continue
    end

    r2 = 1 - residual / ss_tot

    residual_loop[i] = residual
    r2_loop[i] = r2
    slope_loop[i] = slope

    # 🔥 KEY CHANGE: select by slope, not R²
    score = abs(slope - target_slope)

    if score < best_score
        best_score = score
        best_p = p_try
        best_slope = slope
    end
end

println("Estimated pc ≈ ", best_p)
println("Estimated slope ≈ ", best_slope)

## fit 1/νbar at estimated pc
x = log.(N_loop)
y = log.(rc_N .- best_p)
slope = cov(x, y) / var(x)
plot(x, y, label="data", xlabel="log(N)", ylabel="log(|pc(N)-pc|)", title="Scaling of pc(N) with N")
plot!(x, slope .* x .+ mean(y .- slope .* x), label="Fit: slope = $(round(slope, digits=3))", lw=2)

plot(p_candidates, slope_loop, label="Slope at p", xlabel="p", ylabel="Slope", title="Slope vs p")
plot!(p_candidates, r2_loop, label="R² at p", xlabel="p", ylabel="R²", title="R² vs p", yaxis=:right)

##

function estimate_pc_slope(all_data; rc_N=nothing, Δ=0.02, ngrid=50, target_slope=nothing)

    # --- initial guess for pc ---
    if rc_N === nothing
        error("Please provide rc_N (pseudo-critical estimates)")
    end

    pc_guess = mean(rc_N)
    p_candidates = range(pc_guess - Δ, pc_guess + Δ, length=ngrid)

    # storage
    residual_loop = zeros(length(p_candidates))
    r2_loop = fill(NaN, length(p_candidates))
    slope_loop = fill(NaN, length(p_candidates))

    best_p = nothing
    best_slope = nothing

    if target_slope === nothing
        # no prior: choose most negative slope
        best_score = Inf
    else
        # with prior: match slope
        best_score = Inf
    end

    # --- main loop ---
    for (i, p_try) in enumerate(p_candidates)

        N_vals, P_vals, _ = get_vals_at_p(p_try, all_data)

        idx = sortperm(N_vals)
        N_sub = N_vals[idx]
        P_sub = P_vals[idx]

        # remove tiny values
        mask = P_sub .> 1e-12
        if sum(mask) < 3
            continue
        end

        logN = log.(N_sub[mask])
        logP = log.(P_sub[mask])

        slope = cov(logN, logP) / var(logN)

        fit = slope .* logN .+ mean(logP .- slope .* logN)

        residual = sum((logP .- fit).^2)

        y_mean = mean(logP)
        ss_tot = sum((logP .- y_mean).^2)

        if ss_tot < 1e-12
            continue
        end

        r2 = 1 - residual / ss_tot

        residual_loop[i] = residual
        r2_loop[i] = r2
        slope_loop[i] = slope

        # --- selection ---
        if target_slope === nothing
            # general case: pick most negative slope
            score = slope
        else
            # match expected exponent
            score = abs(slope - target_slope)
        end

        if score < best_score
            best_score = score
            best_p = p_try
            best_slope = slope
        end
    end

    return (
        pc = best_p,
        slope = best_slope,
        p_candidates = p_candidates,
        residual = residual_loop,
        r2 = r2_loop,
        slope_curve = slope_loop
    )
end


## FSS p = pc + A N^{-1/νbar}
mask = N_loop .>= 10000 .&& rc_N .> 0
estimate_rc(N_loop, rc_N)
rc = best_p
slope = best_slope
#plot data and fit
scatter(N_loop[mask], rc_N[mask] .- rc, xscale=:log10, yscale=:log10, marker=:circle, xlabel="N", ylabel="pc(N)", title="Estimated pc vs N for SPP", label="Data",markersize=std_Rc_N[mask].*2000)
plot!(N_loop[mask], exp.(slope .* log.(N_loop[mask]) .+ mean(log.(rc_N[mask] .- rc) .- slope .* log.(N_loop[mask]))), label="Fit: pc + A N^{-$(round(-slope, digits=3))}", lw=2, xscale=:log10, yscale=:log10)

## 
gr()

default(
    lw = 2,
    ms = 4,
    framestyle = :box,
    grid = false,
    guidefontsize = 11,
    tickfontsize = 9,
    legendfontsize = 9,   # 🔽 smaller legend
)

# ---------------- Format exponents ----------------
βν_str = @sprintf("%.3f ± %.3f", -slope, std_err)
γν_str = @sprintf("%.3f ± %.3f", slope_chi, std_err_chi)
νbar_str = @sprintf("%.3f", res.nu1bar)

# ---------------- Log-spaced N for fits ----------------
Nfine = exp.(range(log(minimum(N_loop[mask])),
                   log(maximum(N_loop[mask])),
                   length=400))

# ---------------- (a) pc scaling ----------------
plt_pc = plot(legend = :topright)

y_pc = abs.(rc_N[mask] .- res.rc)
yfit_pc = abs.(scaling_model(Nfine, res.fit.param, res.sign) .- res.rc)

scatter!(plt_pc,
    N_loop[mask],
    y_pc,
    xscale = :log10,
    yscale = :log10,
    label = ""
)

plot!(plt_pc,
    Nfine,
    yfit_pc,
    label = L"|p_c(N)-p_c| \sim N^{-1/\bar{\nu}},\; \bar{\nu} = %$νbar_str",
)

using LaTeXStrings
xlabel!(plt_pc, L"N")
ylabel!(plt_pc, L"|p_c(N)-p_c|")

# ---------------- (b) Sc scaling ----------------
plt_S = plot(legend = :bottomleft)

fit_S = exp.(slope .* log.(n_N) .+ mean(y .- slope .* x))

scatter!(plt_S,
    n_N,
    Sc_N,
    yerror = std_Sc_N,
    xscale = :log10,
    yscale = :log10,
    label = ""
)

plot!(plt_S,
    n_N,
    fit_S,
    label = L"S_c \sim N^{-\beta/\bar{\nu}},\; \beta/\bar{\nu} = %$βν_str"
)

xlabel!(plt_S, L"N")
ylabel!(plt_S, L"S_c(N)")

# ---------------- (c) chi scaling ----------------
plt_chi = plot(legend = :bottomright)

fit_chi = exp.(slope_chi .* log.(n_N) .+ mean(y_chi .- slope_chi .* x))

scatter!(plt_chi,
    n_N,
    chi_N,
    yerror = std_chi_N,
    xscale = :log10,
    yscale = :log10,
    label = ""
)

plot!(plt_chi,
    n_N,
    fit_chi,
    label = L"\chi_c \sim N^{\gamma/\bar{\nu}},\; \gamma/\bar{\nu} = %$γν_str"
)

xlabel!(plt_chi, L"N")
ylabel!(plt_chi, L"\chi_c(N)")

# ---------------- Combine panels ----------------
final_plot = plot(
    plt_S, plt_chi,
    layout = (1,2),
    size = (900, 360),   # 🔼 slightly larger figure
    margin = 6Plots.mm
)
final_plot
# savefig(final_plot, "Path_Based_Percolation_ARS/Plot_paper/Estimate_criticalexp_$(model[i])_B$(beta_loop[j]).png")




