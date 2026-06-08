using Plots, Statistics

## High T C = 3
N = [
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    2^16,
    2^17,
    2^18,
    2^19,
    2^20,
    2^21,
    2^22

]

T = [
    6.1,
    12.1,
    32.6,
    109.68,
    414.7,
    1554.0,
    6020.0,
    23593.0,
    93785.0,
    154768.0,
    141920.0,
    592534.0,
    490641.0
]

trials = [300, 300, 300, 300, 300, 300, 300,300,300, 125, 25, 25, 5]

## HighT crossover
N = 2 .^ [10,11,12,13,14,15,16,17,18,19,20,21,22]

T = [4.92,10.01,26.95,95.64,364,
     1448,5889,24937,104780,
     34729,157086,705202, 561779.4]

trials = [300,300,300,300,300,
          300,300,300,300,
          25,25,25,5]

Ttrial = T ./ trials

## Lowt Crossover

N = 2 .^ [10,11,12, 13,14,15,16,17,18,19,20,21,22]

T = [5.7,

     10.7,

     27.17,

     103,

     376,

     1516.17,

     7912,

     30009,

     126898,

     42925,

     134305,

     606077]

trials = [300,300,300,300,300,300,

          300,300,300,

          25,25,25]


## C= N
N = 2 .^ [13,14,15,16,17,18]
T = [98,358,1553,6710,27844,105354]
trials = [300,300,300,300,300,300]

##

logN = log.(N)
logT = log.(T ./ trials)  # normalize by number of trials to get average time per trial

# linear fit: logT = a + b logN
X = [ones(length(logN)) logN]

coef = X \ logT

intercept = coef[1]
slope = coef[2]

println("Estimated slope = ", slope)

Tfit = exp(intercept) .* N.^slope

plt = scatter(
    N, T ./ trials,
    xscale=:log10,
    yscale=:log10,
    xlabel="N",
    ylabel="Running Time (s)",
    label="Data"
)

plot!(
    plt,
    N,
    Tfit,
    lw=2,
    label="Fit: T ∼ N^$(round(slope,digits=3))"
)

display(plt)

# extraploate to N=10^6
N_extrap = 2 .^[19,20,21]
T_extrap = exp(intercept) .* N_extrap.^slope
println("Estimated time for N= $N_extrap: ", T_extrap, " seconds (~", round.(T_extrap/3600, digits=2), " hours)")



