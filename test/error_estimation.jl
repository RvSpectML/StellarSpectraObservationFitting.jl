if !SKIP_SLOW; @testset "estimate_σ_curvature smoke test" begin
    d = _make_tiny_wobble_data(30, 4)
    om = SSOF.OrderModel(d; n_comp_tel=1, n_comp_star=1, oversamp=false)
    mws = SSOF.TotalWorkspace(om, d)
    SSOF.improve_model!(mws; iter=20, verbose=false)
    rvs_out, rvs_σ, tel_s_σ, star_s_σ = SSOF.estimate_σ_curvature(mws)
    @test all(isfinite, rvs_σ)
    @test all(rvs_σ .> 0)
    println()
end; end  # if !SKIP_SLOW
