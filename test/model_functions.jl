@testset "_eval_lm correctness" begin
    M = [1.0 0.0; 0.0 1.0; 1.0 1.0]  # 3×2
    s = [2.0 3.0; 4.0 5.0]            # 2×2
    μ = [10.0, 20.0, 30.0]

    # typed dispatch (M::AbstractMatrix, s::AbstractMatrix, μ::AbstractVector) — always linear
    @test SSOF._eval_lm(M, s, μ) ≈ M * s .+ μ

    # keyword dispatch, log_lm=false
    @test SSOF._eval_lm(M, s, μ; log_lm=false) ≈ M * s .+ μ

    # keyword dispatch, log_lm=true
    @test SSOF._eval_lm(M, s, μ; log_lm=true) ≈ exp.(M * s) .* μ

    # FullLinearModel dispatch
    lm_lin = SSOF.FullLinearModel(M, s, μ, false)
    @test SSOF._eval_lm(lm_lin) ≈ M * s .+ μ

    lm_log = SSOF.FullLinearModel(M, s, μ, true)
    @test SSOF._eval_lm(lm_log) ≈ exp.(M * s) .* μ
    println()
end

@testset "total_model" begin
    tel  = [0.9, 0.8, 0.7]
    star = [1.0, 1.1, 1.2]
    @test SSOF.total_model(tel, star) ≈ tel .* star

    rv = [0.01, 0.02, 0.03]
    @test SSOF.total_model(tel, star, rv) ≈ tel .* (star .+ rv)
    println()
end

@testset "remove_lm_score_means!" begin
    M = rand(4, 2)
    s = rand(2, 5) .+ 1.0   # non-zero mean scores
    μ = rand(4)
    lm = SSOF.FullLinearModel(copy(M), copy(s), copy(μ), false)

    before = SSOF._eval_lm(lm)
    SSOF.remove_lm_score_means!(lm)

    # scores should now have zero mean along the epoch axis
    @test isapprox(mean(lm.s; dims=2), zeros(2, 1); atol=1e-12)
    # the linear model evaluation is invariant to score recentering
    @test isapprox(SSOF._eval_lm(lm), before; rtol=1e-10)
    println()
end

@testset "flip_feature_vectors!" begin
    M = rand(5, 2) .- 0.5
    s = rand(2, 4)
    μ = rand(5)
    lm = SSOF.FullLinearModel(copy(M), copy(s), copy(μ), false)

    # flipping preserves the M*s product (signs cancel column-wise)
    before_product = copy(M) * copy(s)
    SSOF.flip_feature_vectors!(lm)
    @test isapprox(lm.M * lm.s, before_product; rtol=1e-10)

    # flipping twice returns to the same M and s (up to sign consistency)
    M2 = copy(lm.M)
    s2 = copy(lm.s)
    SSOF.flip_feature_vectors!(lm)
    @test isapprox(lm.M, M2; rtol=1e-10)
    @test isapprox(lm.s, s2; rtol=1e-10)
    println()
end

if !SKIP_SLOW; @testset "custom spectra_interp() sensitivity" begin

    B = rand(3,5)
    As = [sparse(rand(2,3)) for i in axes(B, 2)]
    C = rand(5,6)

    # Flat-vector wrapper so EnzymeBackend (AbstractVector path) can be used
    B_vec = copy(vec(B))
    f_flat(xv) = sum(SSOF.spectra_interp(reshape(xv, size(B)) .^ 2, As) * C)

    numer = est_∇(f_flat, copy(B_vec); dif=1e-7)

    # Analytic gradient via Enzyme (exercises the native EnzymeRule for spectra_interp)
    cache = SSOF.prepare_gradient(SSOF.EnzymeBackend(), f_flat, copy(B_vec))
    _, ∂B_flat = SSOF.value_and_gradient!(cache, f_flat, copy(B_vec))

    @test isapprox(∂B_flat, numer; rtol=1e-4)

    println()
end; end  # if !SKIP_SLOW

if !SKIP_SLOW; @testset "EnzymeBackend flat-vector path (spectra_interp via SIH)" begin
    # Exercises the flat-Vector{Float64} EnzymeBackend path against a loss that
    # captures a StellarInterpolationHelper — representative of the closures
    # built by the optimizer paths. FD is the gold-standard correctness gate;
    # Mooncake↔Enzyme agreement is the secondary check.
    n_model, n_obs, n_epochs = 20, 8, 3
    log_λ_obs = collect(LinRange(1.0, 2.0, n_obs)) .+ 0.0 .* (1:n_epochs)'
    log_λ_obs = collect(reshape(log_λ_obs, n_obs, n_epochs))
    model_log_λ = range(0.5, 2.5; length=n_model)
    rvs = randn(n_epochs) .* 0.01
    sih = SSOF.StellarInterpolationHelper(model_log_λ, rvs, log_λ_obs)

    # Flat-vector loss: model_flux flattened to a Vector{Float64}.
    n_flux = n_model * n_epochs
    function ℓ(x::AbstractVector{Float64})
        M = reshape(x, n_model, n_epochs)
        return sum(SSOF.spectra_interp(M, rvs, sih) .^ 2)
    end

    x0 = collect(LinRange(0.5, 1.5, n_flux))

    # Finite-difference reference
    fd = est_∇(ℓ, copy(x0); dif=1e-6)

    # Enzyme (flat-vector path)
    ecache = SSOF.prepare_gradient(SSOF.EnzymeBackend(), ℓ, copy(x0))
    val_en, ∂_en = SSOF.value_and_gradient!(ecache, ℓ, copy(x0))

    @test isapprox(∂_en, fd; rtol=1e-3)

    println()
end; end  # if !SKIP_SLOW
