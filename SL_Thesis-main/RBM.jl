using LinearAlgebra, Plots, Distributions, Random, StableRNGs, Statistics, StatsKit, DataFrames, GLM, LsqFit

include("utilities.jl")

mutable struct RBM 
    N::Int64            # N visible units
    K::Int64            # N hidden units
    α::Float64          # Learning rate
    q::Float64          # Desired sparsity at hidden layer
    r::Float64          # Desired sparsity of connectivity
    λ::Float64          # Weight decay 
    γ::Float64          # Sparsity penalty 
    ρ::Float64          # Momentum weight 
    σ::Float64          # Weight initialization standard deviation
    seed::Int64         # RNG seed
    W::Matrix{Float64}  # Weight matrix 
    M::Matrix{Float64}  # Connectivity mask
    Δ::Matrix{Float64}  # Update matrix
    Δ₀::Matrix{Float64} # Copy (for momentum)

    function RBM(N,K,α,q,r,λ,γ,ρ,σ,seed) 
        rng = StableRNG(seed)
        W = rand(rng, Normal(0, σ), K+1, N+1)
        M = rand(rng, Binomial(1, r), size(W))
        Δ = zeros(size(W))
        return new(N,K,α,q,r,λ,γ,ρ,σ,seed,W,M,Δ,Δ)
    end
end

"""
    up()
Upward pass through RBM
"""
function up(rbm, X::Matrix)
    h = X*rbm.W' |> sigmoid
    h[:,1] = ones(size(h, 1), 1)
    return h
end


"""
    up_nobias()
Upward pass through RBM (Old version that doesnt add bias)
"""
function up_nobias(rbm, X::Matrix)
    return X*rbm.W' |> sigmoid
end


"""
    down()
Downward pass through RBM
"""
function down(rbm, h::Matrix)
    return h*rbm.W |> sigmoid 
end


""" 
    cd1()
Runs an interation of contrastive divergence. 
"""
function cd1(rbm, X::Matrix)
    h = up(rbm, X)
    v = down(rbm, h)
    rbm.Δ = rbm.α.*(h'*X-h'*v) .- rbm.λ.*rbm.W .- rbm.γ*(mean(h)-rbm.q)
    rbm.W += rbm.M .* (rbm.Δ + rbm.ρ .* rbm.Δ₀)
    rbm.Δ₀ = rbm.Δ
    return rbm
end

"""
    train()
Do training run.
"""
function train(rbm::RBM, X::Matrix, X′::Matrix, Y::Vector, Y′::Vector, rng=StableRNG(1235))
    train_idx = reshape(collect(1:length(Y)), length(findall(x->x==1, Y)), length(unique(Y)))
    n_mb = size(train_idx, 1)
    shuffled_indices = shuffle(rng, collect(1:n_mb))
    train_idx = train_idx[shuffled_indices]
    
    res = zeros(n_mb, 3)
    for i ∈ 1:n_mb
        x = X[train_idx[i,:],:]

        # Contrastive Divergence (1)
        rbm = cd1(rbm, x)
        
        # Compute train set error 
        h = up(rbm, x)
        v = down(rbm, h)

        # Compute test set error
        h′ = up(rbm, X′)
        v′ = down(rbm, h′)

        # Pack results
        res[i,1] = norm(rbm.Δ)
        res[i,2] = abs.(x .- v) |> mean
        res[i,3] = abs.(X′ .- v′) |> mean
    end 

    return rbm, res[2:end,:]
end

function shuffle_features(X::Matrix, seed=1234)
    rng = StableRNG(seed)
    N, K = size(X)
    for i ∈ 1:N 
        X[i, 2:K] .= shuffle(rng, X[i, 2:K])
    end
    return X
end


"""
    energy()
Energy of a network after an up/down pass
"""
function energy(h::Matrix, v::Matrix, W::Matrix)
    Xn , = size(v)
    E = []
    for i in 1:Xn
        append!(E, -v[i,:]' * W' * h[i,:])
    end
    return E
end

"""
    energy_patterns()
Energy for each pattern
"""
function energy_patterns(X::Matrix, DG::RBM)
    h = up(DG, X)
    v = down(DG, h)
    E = energy(h, v, DG.W)
    return E
end

"""
    similarity()
Returns similarity of two patterns
"""
function similarity(a::Vector{Float64}, b::Vector{Float64})
    return sum(abs.(a .- b))
end

"""
    fit_power_law(x, y)
Fits `a + (1-a)*(x^b)` to the pattern separation data. 
"""

function fit_power_law(x::Vector, y::Vector)
    # Clip for tractability
    x = max.(x, 0.0001)
    y = max.(y, 0.0001)

    @. model(x, w) = w[1] + (1-w[1])*(x^w[2])
    res = curve_fit(model, x, y, [0., 1.])
    a, b = res.param 
    f(x) = a + (1-a)*(x^b)
    auc = -(2*a*b - b + 1) / (2*(1+b))
    return f, auc
end

""" 
    pattern_separation()
Computes pairwise correlations between input and output patterns.

Returns area under the curve, figure of the curve, pattern seperation curve, and vectors of pairwise correlations of perforant pathway (PP) and granule cells (GC)
"""

function pattern_separation(X::Matrix, DG::RBM)
    h = up(DG, X)

    PPcor = Vector{Float32}(undef, 0)
    GCcor = Vector{Float32}(undef, 0)

    for i in 1:size(X, 1)
        for j in (i+1):size(X, 1) #for each combination of patterns
            
            PPcor = append!(PPcor, cor(X[i,:], X[j,:]))
            GCcor = append!(GCcor, cor(h[i,:], h[j,:]))
        end
    end

    fpl_curve, area_under_curve = fit_power_law(PPcor, GCcor) #get pattern separation curve

    figPS = plot(PPcor, GCcor, seriestype=:scatter, xlabel="PP correlations", ylabel="GC correlations")
    x = range(0, 1, length=100)
    y = fpl_curve.(x)
    figPS = plot!(x,y)

    return area_under_curve, figPS, fpl_curve, PPcor, GCcor

end


""" 
    mnemonic_discrimination()
Computes mnemonic discrimination capacity of an RBM given X patterns

Returns slope of energy x edit distance plot, the figure of said plot, energy vector, and edit distance vector
"""

function mnemonic_discrimination(X::Matrix, DG::RBM)

    Eall = Float64.(energy_patterns(X, DG))

    Sim_all = Vector{Float64}(undef, 0)

    for i in 1:size(X, 1)
        S_hold = Vector{Float64}(undef, 0)
        for j in 1:size(X, 1)
            if i != j
                S_hold = append!(S_hold, similarity(X[i,:], X[j,:]))
            end
        end

        Sim_all = append!(Sim_all, mean(S_hold))
    end

    figMD = scatter(Sim_all, Eall, xlabel="Average Edit Distance", ylabel="Energy")

    data = DataFrame(Energy = Eall, Edit_Distance = Sim_all)
    fm = @formula(Energy ~ Edit_Distance)
    linearRegressor = lm(fm, data)

    slope = coef(linearRegressor)[2]

    return slope, figMD, Eall, Sim_all
end