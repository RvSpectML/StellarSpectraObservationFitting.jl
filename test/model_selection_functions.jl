@testset "ℓ_prereqs, ℓ, aic, aicc, bic" begin
    # ℓ_prereqs returns (logdet_Σ, n)
    vars = [1.0 2.0; 3.0 Inf]
    logdet_Σ, n = SSOF.ℓ_prereqs(vars)
    @test n == 3
    @test isapprox(logdet_Σ, log(1.0) + log(2.0) + log(3.0); rtol=1e-10)

    # ℓ: zero chi-squared, zero logdet_Σ, n=1 → -0.5*log(2π)
    @test isapprox(SSOF.ℓ(0.0, 0.0, 1), -0.5 * log(2π); rtol=1e-10)

    # aic scalar form: 2*(k - ℓ)
    @test SSOF.aic(2, 10.0) ≈ 2 * (2 - 10.0)

    # bic scalar form: k*log(n) - 2*ℓ
    @test SSOF.bic(3, 10.0, 100) ≈ 3 * log(100) - 2 * 10.0

    # aicc: aic + correction term
    @test SSOF.aicc(2, 10.0, 100) ≈ SSOF.aic(2, 10.0) + (2 * 2 * 3) / (100 - 2 - 1)
    println()
end

@testset "intra_night_std" begin
    # single night with ≥ thres identical observations → std = 0
    rvs_flat = [100.0, 100.0, 100.0, 100.0]
    times_single = [0.0, 0.1, 0.2, 0.3]
    @test SSOF.intra_night_std(rvs_flat, times_single; show_warn=false) == 0.0

    # fewer than thres observations in every night → Inf
    rvs_few = [1.0, 2.0]
    times_few = [0.0, 0.1]
    @test SSOF.intra_night_std(rvs_few, times_few; thres=3, show_warn=false) == Inf
    println()
end
