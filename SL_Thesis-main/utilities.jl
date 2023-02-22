using StableRNGs

"""
    Utility Functions
"""
sigmoid(X::Matrix) = 1 ./ (1 .+ exp.(-1 .* X))
bitflip(x) = 2. *x - 1.
bitflip_inv(x) = (x + 1.)/2.

"""
    make_patterns()
Creates the patterns according to Finnegan & Becker (2014).
Arguments: 
    - `n_dim`: Number of dimensions 
    - `n_prototypes`: Number of core prototypes 
    - `p`: Sparsity level 
    - `rng`: Pseudorandom number generator
"""
function make_patterns(; n_dim::Int64=100, n_prototypes::Int64=5, p::Float64=0.1, p_perm1::Float64=0.2, p_perm2::Float64=0.05, 
    rng=StableRNG(1234))
    n_bits_on = round(Int, p * n_dim)
    X = zeros(n_prototypes, n_dim)  # Training
    X′ = zeros(n_prototypes, n_dim) # Testing
    Y = []; Y′ =[] # Class labels
    for i ∈ 1:n_prototypes
        # Generate n_prototypes core classes
        low = n_bits_on*(i-1) + 1 
        high = n_bits_on*i
        X[i,low:high] .= 1
        
        for j ∈ 1:10
            # Generate 10 prototypes per core
            permutation1 = rand(rng, Binomial(1, 1-p_perm1), size(X[i,:])) .* 1.0
            x = bitflip_inv.(bitflip.(permutation1).*bitflip.(X[i,:]))

            # For each of the 10 prototypes in a core, generate 24 permutations
            for k ∈ 1:24 
                permutation2 = rand(rng, Binomial(1, 1-p_perm2), size(x)) .* 1.0
                Z = bitflip_inv.(bitflip.(permutation2).*bitflip.(x))
                if k < 21
                    X = [X; Z']     # Add to training set
                    append!(Y, i)
                else 
                    X′ = [X′; Z']   # Add to testing set
                    append!(Y′, i)
                end
            end
        end
    end
    
    # Remove core prototypes and add bias terms 
    X, X′ = X[n_prototypes+1:end,:], X′[n_prototypes+1:end,:]
    X, X′ = [ones(size(X, 1), 1) X], [ones(size(X′, 1), 1) X′]
    return X, X′, Y, Y′   
end


"""
    make_patterns_novel_tests()
Creates the patterns according to Finnegan & Becker (2014). Creates exclusive categories for the test set
Arguments: 
    - `n_dim`: Number of dimensions 
    - `n_prototypes`: Total number of core prototypes 
    - `n_prototypes_new`: Number of prototypes exclusive to testing set
    - `p_test`: proportion of patterns for each category to be allocated to the test set
    - `p`: Sparsity level 
    - `rng`: Pseudorandom number generator
"""
function make_patterns_novel_tests(; n_dim::Int64=100, n_prototypes::Int64=7, n_prototypes_new::Int64=2,p_test::Float64 = 1/6, p::Float64=0.1, p_perm1::Float64=0.2, p_perm2::Float64=0.05, 
    seed=1234)
    rng = StableRNG(seed)
    n_bits_on = round(Int, p * n_dim)
    X = zeros(n_prototypes, n_dim)  # Training
    X′ = zeros(n_prototypes, n_dim) # Testing
    Y = []; Y′ =[] # Class labels
    k_train = round(Int, p_test * 24)
    for i ∈ 1:n_prototypes
        # Generate n_prototypes core classes
        low = n_bits_on*(i-1) + 1 
        high = n_bits_on*i
        X[i,low:high] .= 1

        for j ∈ 1:10
            # Generate 10 prototypes per core
            permutation1 = rand(rng, Binomial(1, 1-p_perm1), size(X[i,:])) .* 1.0
            x = bitflip_inv.(bitflip.(permutation1).*bitflip.(X[i,:]))

            # For each of the 10 prototypes in a core, generate 24 permutations
            for k ∈ 1:24 
                permutation2 = rand(rng, Binomial(1, 1-p_perm2), size(x)) .* 1.0
                Z = bitflip_inv.(bitflip.(permutation2).*bitflip.(x))
                if k <= 24 - k_train && i <= n_prototypes - n_prototypes_new
                    X = [X; Z']     # Add to training set
                    append!(Y, i)
                elseif k > 24 - k_train
                    X′ = [X′; Z']   # Add to testing set
                    append!(Y′, i)
                end
            end
        end
        
        
    end
    
    # Remove core prototypes and add bias terms 
    X, X′ = X[n_prototypes+1:end,:], X′[n_prototypes+1:end,:]
    X, X′ = [ones(size(X, 1), 1) X], [ones(size(X′, 1), 1) X′]
    return X, X′, Y, Y′   
end

"""
    combine_test_train()
Returns combination of test and training patterns with label vector (0 = trainset, 1 = testset with same categories, 2 = testset with new category)
Arguments: 
- `X`: train set patterns
- `Xt`: test set patterns
- `p_excl`: proportion of training set patterns that have an exclusive category
"""

function combine_test_train(X::Matrix, Xt::Matrix, p_excl::Float64)

    Xall = vcat(X, Xt)
    X_n ,  = Int.(size(X))
    Xt_n ,  = Int.(size(Xt))
    Xall_n ,  = Int.(size(Xall))

    Xall_labs = zeros(Xall_n)
    Xall_labs[X_n+1:Int((1-p_excl)*Xt_n + X_n)] .= 1 #1 if part of test set
    Xall_labs[Int((1-p_excl)*Xt_n + X_n)+1:end] .= 2 #2 if contains exclusive category

    return Xall, Xall_labs
end

