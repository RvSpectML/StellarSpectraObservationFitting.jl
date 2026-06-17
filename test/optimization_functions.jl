@testset "Task 3a: opt_funcs gradient correctness with EnzymeBackend" begin
    # Verifies that loss_funcs_telstar_v2 (inline Tuple path) gives Enzyme gradients
    # that agree with finite differences. The original l_telstar ∘ unflatten path is
    # used only as the FD reference (nested Vector{Any} loses Enzyme activity tracking).
    # Perturbation: M=0, s=0, μ=1 are L1-subdifferential corners where sign(0)=0
    # (Enzyme) ≠ right-side FD derivative (±L1_coeff). Perturb to differentiable point.
    d = _make_tiny_wobble_data(20, 4)
    om = SSOF.OrderModel(d; n_comp_tel=1, n_comp_star=1, oversamp=false)
    om.tel.lm.M  .= 0.01 .* randn(size(om.tel.lm.M))
    om.tel.lm.s  .= 0.1  .* randn(size(om.tel.lm.s))
    om.tel.lm.μ  .+= 0.1  .* randn(size(om.tel.lm.μ))
    om.star.lm.M .= 0.01 .* randn(size(om.star.lm.M))
    om.star.lm.s .= 0.1  .* randn(size(om.star.lm.s))
    om.star.lm.μ .+= 0.1  .* randn(size(om.star.lm.μ))
    o = SSOF.Output(om, d)

    l_v2 = SSOF.loss_funcs_telstar_v2(o, om, d)
    θ = (SSOF._lm_tuple(om.tel.lm), SSOF._lm_tuple(om.star.lm))

    cache_en = SSOF.prepare_gradient(SSOF.EnzymeBackend(), l_v2, θ)
    val_en, g_en = SSOF.value_and_gradient!(cache_en, l_v2, θ)

    # FD reference via mathematically equivalent l_telstar ∘ unflatten
    pars = [vec(om.tel.lm), vec(om.star.lm)]
    p0, unflatten = SSOF.flatten(pars)
    l_telstar, _, _ = SSOF.loss_funcs_telstar(o, om, d)
    fd = est_∇(l_telstar ∘ unflatten, copy(p0); dif=1e-5)

    @test isfinite(val_en)
    # Flatten Enzyme gradient in same order as ParameterHandling.flatten:
    # tel arrays (M, s, μ) then star arrays (M, s, μ), each column-major.
    g_en_flat = vcat(vec.(g_en[1])..., vec.(g_en[2])...)
    @test isapprox(g_en_flat, fd; rtol=1e-3)

    println()
end

@testset "Task 3b: improve_model! end-to-end smoke test with EnzymeBackend" begin
    # Checks that improve_model! on a TotalWorkspace (EnzymeBackend default) decreases
    # the loss and leaves finite RVs — exercises the full Adam + finalize_scores! path.
    d = _make_tiny_wobble_data(30, 4)
    om = SSOF.OrderModel(d; n_comp_tel=1, n_comp_star=1, oversamp=false)
    mws = SSOF.TotalWorkspace(om, d)  # EnzymeBackend() by default

    loss_before = SSOF._loss(mws)
    SSOF.improve_model!(mws; iter=20, verbose=false)
    loss_after = SSOF._loss(mws)

    @test loss_after < loss_before
    @test all(isfinite, mws.om.rv)

    println()
end

@testset "Task 3c: LSFData Enzyme gradient through spectra_interp sparse rule" begin
    # Exercises the Enzyme rule for spectra_interp with a SparseMatrixCSC (d.lsf).
    # Uses loss_funcs_telstar_v2 (Tuple path) so Enzyme correctly tracks all active
    # parameters. d.lsf is captured Const; gradient flows through lsf' * ∂Y.
    n_obs = 15
    n_epochs = 3
    log_λ_obs = collect(LinRange(8.78535, 8.78590, n_obs)) .+ 1e-5 .* (0:n_epochs-1)'
    log_λ_star = log_λ_obs
    flux = ones(n_obs, n_epochs) .+ 0.05 .* randn(n_obs, n_epochs)
    flux = max.(flux, 1e-6)
    var = fill(1e-4, n_obs, n_epochs)
    lsf = spdiagm(0 => fill(0.6, n_obs), -1 => fill(0.2, n_obs-1), 1 => fill(0.2, n_obs-1))
    d = SSOF.LSFData(flux, var, var, log_λ_obs, log_λ_star, lsf)
    om = SSOF.OrderModel(d; n_comp_tel=1, n_comp_star=1, oversamp=false)
    # Perturb away from L1 subdifferential corners (same reasoning as Task 3a)
    om.tel.lm.M  .= 0.01 .* randn(size(om.tel.lm.M))
    om.tel.lm.s  .= 0.1  .* randn(size(om.tel.lm.s))
    om.tel.lm.μ  .+= 0.1  .* randn(size(om.tel.lm.μ))
    om.star.lm.M .= 0.01 .* randn(size(om.star.lm.M))
    om.star.lm.s .= 0.1  .* randn(size(om.star.lm.s))
    om.star.lm.μ .+= 0.1  .* randn(size(om.star.lm.μ))
    o = SSOF.Output(om, d)

    l_v2 = SSOF.loss_funcs_telstar_v2(o, om, d)
    θ = (SSOF._lm_tuple(om.tel.lm), SSOF._lm_tuple(om.star.lm))

    cache_en = SSOF.prepare_gradient(SSOF.EnzymeBackend(), l_v2, θ)
    val_en, g_en = SSOF.value_and_gradient!(cache_en, l_v2, θ)

    # FD reference via mathematically equivalent l_telstar ∘ unflatten
    pars = [vec(om.tel.lm), vec(om.star.lm)]
    p0_flat, unflatten = SSOF.flatten(pars)
    l_telstar, _, _ = SSOF.loss_funcs_telstar(o, om, d)
    fd = est_∇(l_telstar ∘ unflatten, copy(p0_flat); dif=1e-6, inds=1:min(10, length(p0_flat)))

    g_en_flat = vcat(vec.(g_en[1])..., vec.(g_en[2])...)
    @test isapprox(g_en_flat[1:min(10, length(p0_flat))], fd; rtol=1e-3)

    println()
end

if !SKIP_SLOW; @testset "calculate_initial_model smoke test" begin
    # Exercises the full model-selection pipeline (EMPCA init, AIC comparison across
    # n_comp combinations, improve_model! for each candidate) on a tiny dataset.
    d = _make_tiny_wobble_data(30, 6)
    times = collect(1.0:6.0)
    om = SSOF.calculate_initial_model(d; max_n_tel=1, max_n_star=1, oversamp=false, times=times)
    @test om isa SSOF.OrderModelWobble
    @test all(isfinite, om.star.lm.μ)
    @test all(isfinite, om.tel.lm.μ)
    @test all(isfinite, om.rv)
    println()
end; end  # if !SKIP_SLOW

if !SKIP_SLOW; @testset "EnzymeBackend nested-tuple θ for OrderModelDPCA (l_total)" begin
    # Build a minimal DPCA model. l_total for DPCA inlines the spectral computation
    # (tel_o, star_o, rv_o via positional args) so Enzyme sees all active variables
    # through positional arguments rather than kwargs, ensuring correct activity tracking.
    n_obs = 20
    n_epochs = 4
    log_λ_obs = collect(LinRange(8.78535, 8.78602, n_obs)) .+ 1e-5 .* (0:n_epochs-1)'
    log_λ_star = log_λ_obs
    flux = ones(n_obs, n_epochs) .+ 0.1 .* randn(n_obs, n_epochs)
    flux = max.(flux, 1e-6)
    var = fill(1e-4, n_obs, n_epochs)
    d = SSOF.GenericData(flux, var, var, log_λ_obs, log_λ_star)
    om = SSOF.OrderModel(d; dpca=true, n_comp_tel=1, n_comp_star=1, oversamp=false)
    # initialize templates on the model grid (length differs from n_obs)
    om.star.lm.μ .= 1 .+ 0.1 .* sin.(om.star.λ)
    om.tel.lm.μ .= one(eltype(om.tel.lm.μ))
    om.rv.lm.M[:, 1] .= SSOF.doppler_component(om.star.λ, om.star.lm.μ)
    o = SSOF.Output(om, d)

    l_total, _ = SSOF.loss_funcs_total(o, om, d)
    θ = (SSOF._lm_tuple(om.tel.lm), SSOF._lm_tuple(om.star.lm), om.rv.lm.s)

    cache_en = SSOF.prepare_gradient(SSOF.EnzymeBackend(), l_total, θ)
    val_en, ∂θ_en = SSOF.value_and_gradient!(cache_en, l_total, θ)

    # gradient w.r.t. star μ must include the doppler-basis path
    ∂μ_en = ∂θ_en[2][3]

    # finite-difference sanity on star μ gradient
    function l_total_flat_μ(μ_flat)
        tel_t, star_t, rv_s = θ
        return l_total((tel_t, (star_t[1], star_t[2], μ_flat), rv_s))
    end
    fd_μ = est_∇(l_total_flat_μ, copy(om.star.lm.μ); dif=1e-6)

    @test isapprox(∂μ_en, fd_μ; rtol=1e-3)

    println()
end; end  # if !SKIP_SLOW
