@testset "fit_continuum polynomial recovery" begin
    x = collect(LinRange(0.0, 1.0, 50))
    true_poly = @. 1.0 + 2.0 * x - x ^ 2
    # small noise to keep std(resid) > 0 for sigma-clipping convergence
    y = true_poly .+ 1e-3 .* randn(50)
    σ² = fill(1e-6, 50)
    μ_fit, _ = SSOF.fit_continuum(x, y, σ²; order=3)
    @test isapprox(μ_fit, true_poly; atol=0.05)
    println()
end

@testset "mask_bad_edges! masks at least one edge with always_mask_something" begin
    d = _make_tiny_wobble_data(30, 4)
    n_finite_before = sum(isfinite, d.var)
    SSOF.mask_bad_edges!(d; verbose=false, always_mask_something=true)
    @test sum(isfinite, d.var) < n_finite_before
    println()
end
