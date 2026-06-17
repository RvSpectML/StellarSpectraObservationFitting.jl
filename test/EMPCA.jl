import ExpectationMaximizationPCA as EMPCA_pkg

@testset "EMPCA! rank-1 recovery" begin
    n, k, m = 12, 1, 25
    M_true = randn(n, k)
    s_true = randn(k, m)
    μ_true = ones(n)
    # near-noiseless rank-1 data
    data = M_true * s_true .+ μ_true .+ 1e-6 .* randn(n, m)
    weights = ones(n, m)

    lm = SSOF.FullLinearModel(randn(n, k), randn(k, m), copy(μ_true), false)
    EMPCA_pkg.EMPCA!(lm, copy(data), copy(weights); niter=50)

    # recovered M[:,1] must be highly correlated with M_true[:,1]
    corr = abs(dot(lm.M[:, 1], M_true[:, 1])) / (norm(lm.M[:, 1]) * norm(M_true[:, 1]))
    @test corr > 0.99
    println()
end
