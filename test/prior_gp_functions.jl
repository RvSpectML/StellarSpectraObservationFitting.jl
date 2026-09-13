@testset "fast GP prior likelihoods (and their gradients)" begin
    x = 8.78535917650598:6.616545829861497e-7:8.786020831088965
    fx = SSOF.SOAP_gp(x)
    y = rand(fx)

    # setting up constants and precalcuating gradient coefficients
    H_k, P∞, σ²_meas = SSOF.H_k, SSOF.P∞, SSOF._σ²_meas_def
    A_k, Σ_k = SSOF.SOAP_gp_sde_prediction_matrices(step(x))
    sparsity = Int(round(0.5 / (step(x) * SSOF.SOAP_gp_params.λ)))
    Δℓ_coe = SSOF.gp_Δℓ_coefficients(length(y), A_k, Σ_k; H_k=H_k, P∞=P∞, σ²_meas=σ²_meas)
    Δℓ_coe_s = SSOF.gp_Δℓ_coefficients(length(y), A_k, Σ_k; H_k=H_k, P∞=P∞, σ²_meas=σ²_meas, sparsity=sparsity)

    f(y) = SSOF.gp_ℓ(y, A_k, Σ_k; σ²_meas=σ²_meas)
    numer = est_∇(f, y; dif=1e-9)
    anal = SSOF.gp_Δℓ(y, A_k, Σ_k, H_k, P∞; σ²_meas=σ²_meas)
    anal_p = SSOF.Δℓ_precalc(Δℓ_coe, y, A_k, Σ_k, H_k, P∞; σ²_meas=σ²_meas)
    anal_p_s = SSOF.Δℓ_precalc(Δℓ_coe_s, y, A_k, Σ_k, H_k, P∞; σ²_meas=σ²_meas)

    @test isapprox(numer, anal; rtol=1e-4)
    @test isapprox(numer, anal_p; rtol=1e-4)
    @test isapprox(numer, anal_p_s; rtol=1e-4)

    println()
end

@testset "steady-state Kalman prior matches the general filter" begin
    H_k, P∞, σ²_meas = SSOF.H_k, SSOF.P∞, SSOF._σ²_meas_def

    # Two grids with very different warm-up lengths (see profiling/GPU_FEASIBILITY.md
    # §3): a fine grid (long warm-up, like SSOF's stellar submodel) and a coarse one
    # (short warm-up, like the telluric submodel).
    for (Δx, n) in ((6.616545829861497e-7, 3000), (5.134684755671457e-6, 3000))
        x = range(8.7853, step=Δx, length=n)
        fx = SSOF.SOAP_gp(x)
        y = rand(fx)

        A_k, Σ_k = SSOF.SOAP_gp_sde_prediction_matrices(step(x))
        ss = SSOF.steady_state_gp(A_k, Σ_k; H_k=H_k, P∞=P∞, σ²_meas=σ²_meas)

        ℓ_gen = SSOF.gp_ℓ(y, A_k, Σ_k; σ²_meas=σ²_meas)
        ℓ_ss = SSOF.gp_ℓ(y, ss)
        @test isapprox(ℓ_ss, ℓ_gen; rtol=1e-12)

        γ_gen = SSOF.gp_Δℓ_helper_γ(y, A_k, Σ_k, H_k, P∞; σ²_meas=σ²_meas)
        γ_ss = SSOF.gp_Δℓ_helper_γ(y, ss)
        @test isapprox(γ_ss, γ_gen; rtol=1e-10)

        sparsity = Int(round(0.5 / (step(x) * SSOF.SOAP_gp_params.λ)))
        Δℓ_coe = SSOF.gp_Δℓ_coefficients(n, A_k, Σ_k; H_k=H_k, P∞=P∞, σ²_meas=σ²_meas, sparsity=sparsity)
        @test isapprox(SSOF.gp_ℓ_precalc(Δℓ_coe, y, ss), ℓ_gen; rtol=1e-12)
        @test isapprox(SSOF.Δℓ_precalc(Δℓ_coe, y, ss), SSOF.Δℓ_precalc(Δℓ_coe, y, A_k, Σ_k, H_k, P∞; σ²_meas=σ²_meas); rtol=1e-10)
    end

    println()
end
