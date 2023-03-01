using LinearAlgebra, Plots, Distributions, Random, StableRNGs, Statistics, StatsKit, HypothesisTests, StatsPlots

include("utilities.jl")
include("RBM.jl")

global seed = 1234

X, Xt, Y, Yt = make_patterns_novel_tests(n_prototypes=7, n_prototypes_new=2,p_test = 1/6)
Xall, Xall_labs = combine_test_train(X, Xt, 2/7)

DG_n = 10 # Sample size per condition
condition_multiplier = [0.2, 0.5, 1, 2, 5] # variations of independent variable (usually [0.2, 0.5, 1, 2, 5])
condition_n, = size(condition_multiplier) # Number of conditions

DGexp = Matrix{RBM}(undef, condition_n, DG_n)
AUC = Matrix{Float64}(undef, condition_n, DG_n)
figsPS = Matrix(undef, condition_n, DG_n)
MD = Matrix{Float64}(undef, condition_n, DG_n)
figsMD = Matrix(undef, condition_n, DG_n)
LDI = Matrix{Float64}(undef, condition_n, DG_n)
REC = Matrix{Float64}(undef, condition_n, DG_n)
Eold = Matrix{Float64}(undef, condition_n, DG_n)
Elures = Matrix{Float64}(undef, condition_n, DG_n)
Efoils = Matrix{Float64}(undef, condition_n, DG_n)

#initial parameter values
N = 100; K = 500
α=0.01; q=0.01; r=0.1; λ=0.005; γ=1; ρ=0.8; σ=0.001


for i in 1:condition_n #for each condition
    global seed = 1234
    for j in 1:DG_n #for each RBM
        #initialization
        DGexp[i,j] = RBM(N, K, α, q, r * condition_multiplier[i], λ, γ, ρ, σ, seed) #set indepenent variable here ( * condition_multiplier[i])
        DGexp[i,j], res_DG = train(DGexp[i,j], X, Xt, Y, Yt)
    
        AUC[i,j], figsPS[i,j], = pattern_separation(Xt, DGexp[i,j])
        MD[i,j], figsMD[i,j], = mnemonic_discriminationV2(Xt, X, DGexp[i,j])
        LDI[i,j], REC[i,j], Eold[i,j], Elures[i,j], Efoils[i,j] = energy_lure_foils(Xall, Xall_labs, DGexp[i,j])
    
        global seed += 1
    end
end

fig_AUC = violin(AUC[1,:], xlabel="Condition multiplier", ylabel="Area under pattern separation curve", xticks = ([1:condition_n;], string.(condition_multiplier)), legend = false)
for i in 2:condition_n
    fig_AUC = violin!(AUC[i,:])
end
fig_MD = violin(MD[1,:], xlabel="Condition multiplier", ylabel="Mnemonic Discrimination", xticks = ([1:condition_n;], string.(condition_multiplier)), legend = false)
for i in 2:condition_n
    fig_MD = violin!(MD[i,:])
end
fig_LDI = violin(LDI[1,:], xlabel="Condition multiplier", ylabel="LDI", xticks = ([1:condition_n;], string.(condition_multiplier)), legend = false)
for i in 2:condition_n
    fig_LDI = violin!(LDI[i,:])
end
fig_REC = violin(REC[1,:], xlabel="Condition multiplier", ylabel="REC", xticks = ([1:condition_n;], string.(condition_multiplier)), legend = false)
for i in 2:condition_n
    fig_REC = violin!(REC[i,:])
end
fig_lures = violin(Elures[1,:], xlabel="Condition multiplier", ylabel="Energy Lures", xticks = ([1:condition_n;], string.(condition_multiplier)), legend = false)
for i in 2:condition_n
    fig_lures = violin!(Elures[i,:])
end
fig_foils = violin(Efoils[1,:], xlabel="Condition multiplier", ylabel="Energy Foils", xticks = ([1:condition_n;], string.(condition_multiplier)), legend = false)
for i in 2:condition_n
    fig_foils = violin!(Efoils[i,:])
end
