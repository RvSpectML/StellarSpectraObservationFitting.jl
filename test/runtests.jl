using Test
import TemporalGPs; TGP = TemporalGPs
using SparseArrays
import StellarSpectraObservationFitting as SSOF
using LinearAlgebra

println("Testing...")

# Set SSOF_SKIP_SLOW_TESTS=true to skip testsets that take >60 s each:
#   "EnzymeBackend flat-vector path (spectra_interp via SIH)"
#   "EnzymeBackend nested-tuple θ for OrderModelDPCA (l_total)"
const SKIP_SLOW = get(ENV, "SSOF_SKIP_SLOW_TESTS", "false") == "true"
SKIP_SLOW && println("Skipping slow tests (SSOF_SKIP_SLOW_TESTS=true)")

@testset "AD backends load" begin
    @test SSOF.MooncakeBackend() isa SSOF.ADBackend
    @test SSOF.EnzymeBackend() isa SSOF.ADBackend
end

function est_∇(f::Function, inputs; dif::Real=1e-7, inds::AbstractUnitRange=eachindex(inputs))
    val = f(inputs)
    grad = Array{Float64}(undef, length(inds))
    for i in inds
        hold = inputs[i]
        inputs[i] += dif
        grad[i] =  (f(inputs) - val) / dif
        inputs[i] = hold
    end
    return grad
end

@testset "fast GP prior likelihoods (and their gradients)" begin
    x = 8.78535917650598:6.616545829861497e-7:8.786020831088965
    fx = SSOF.SOAP_gp(x)
    y = rand(fx)

    # are my likelihood calculations the same as TemporalGPs
    # @test isapprox(TGP.logpdf(fx, y), SSOF.SOAP_gp_ℓ(y, step(x)))
    # @test isapprox(TGP.logpdf(fx, y), SOAP_gp_ℓ_nabla(y, step(x)))

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

    # f(y) = SSOF.gp_ℓ_nabla(y, A_k, Σ_k; σ²_meas=σ²_meas)
    # nabla = only(∇(f)(y_test))
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

@testset "EnzymeBackend nested-tuple θ with aliasing into captured om" begin
    # Phase 3: the Adam path. Production calls look like:
    #   AdamSubWorkspace((om.tel.lm.s, om.star.lm.s, om.rv), l_total)
    # where each θ entry is identity-equal to an array inside `om`, and the
    # loss closure captures `om`. The shadow ∂θ must alias ∂l's captured ∂om
    # in the same way. Tested here with a minimal synthetic struct that mimics
    # the relevant aspects.
    mutable struct MiniOM
        M::Matrix{Float64}
        s::Vector{Float64}
        μ::Vector{Float64}
        bias::Matrix{Float64}    # captured-only context (not in θ)
    end
    om = MiniOM(rand(4, 3), rand(3), rand(4), rand(4, 5))

    # Aliasing invariant: θ[1] === om.M, θ[2] === om.s, θ[3] === om.μ.
    # The loss reads BOTH θ AND om.bias (captured); correctness requires
    # gradients to flow through θ even though the same arrays are reachable
    # via captured om.
    build_θ_minimini(om) = (om.M, om.s, om.μ)
    build_l_minimini(om, _, _) = function (θ)
        # Reads θ AND captured om.bias; computes loss = ||θ[1]*θ[2]+θ[3] − col_means(om.bias)||²
        pred = θ[1] * θ[2] .+ θ[3]
        target = vec(sum(om.bias; dims=2)) ./ size(om.bias, 2)
        return sum((pred .- target) .^ 2)
    end

    θ = build_θ_minimini(om)
    l = build_l_minimini(om, nothing, nothing)

    # Aliasing assertions on the primal (sanity check that the test setup is right)
    @test pointer(θ[1]) === pointer(om.M)
    @test pointer(θ[2]) === pointer(om.s)
    @test pointer(θ[3]) === pointer(om.μ)

    # Enzyme cache
    cache = SSOF.prepare_gradient(
        SSOF.EnzymeBackend(), l, θ;
        om=om, build_θ=build_θ_minimini, build_l=build_l_minimini,
        o=nothing, d=nothing,
    )

    val_en, ∂θ_en = SSOF.value_and_gradient!(cache, l, θ)

    # FD reference: differentiate the closure w.r.t. a flat parameterisation
    function ℓ_flat(x)
        M = reshape(view(x, 1:12),   4, 3)
        s = view(x, 13:15)
        μ = view(x, 16:19)
        pred = M * s .+ μ
        target = vec(sum(om.bias; dims=2)) ./ size(om.bias, 2)
        return sum((pred .- target) .^ 2)
    end
    x_flat = vcat(vec(om.M), om.s, om.μ)
    fd = est_∇(ℓ_flat, copy(x_flat); dif=1e-6)

    ∂_flat = vcat(vec(∂θ_en[1]), ∂θ_en[2], ∂θ_en[3])
    @test isapprox(val_en, ℓ_flat(x_flat); rtol=1e-10)
    @test isapprox(∂_flat, fd; rtol=1e-3)

    # Second call uses the same cache; verify remake_zero! correctly resets
    val_en2, ∂θ_en2 = SSOF.value_and_gradient!(cache, l, θ)
    @test isapprox(val_en2, val_en; rtol=1e-12)
    @test isapprox(∂θ_en2[1], ∂θ_en[1]; rtol=1e-12)

    println()
end

@testset "EnzymeBackend on loss ∘ unflatten composition" begin
    # opt_funcs (optimization_functions.jl:804) differentiates `f = loss ∘ unflatten`
    # where unflatten rebuilds a nested parameter structure from a flat vector.
    # Phase 5 will switch the default backend to Enzyme via this exact composition,
    # so Phase 2's done-gate must exercise it — a hand-written ℓ on a flat input
    # does not capture the closure shape (the unflatten closure carries its own
    # captured shape info) that production code uses.
    pars0 = [rand(3, 2), rand(2, 3), rand(5)]
    loss(pars) = sum(pars[1]) + sum(pars[2] .^ 2) + sum(pars[3] .^ 3)

    p0, unflatten = SSOF.flatten(pars0)
    f = loss ∘ unflatten

    cache_en = SSOF.prepare_gradient(SSOF.EnzymeBackend(), f, copy(p0))
    val_en, ∂_en = SSOF.value_and_gradient!(cache_en, f, copy(p0))

    fd = est_∇(f, copy(p0); dif=1e-6)

    @test isapprox(∂_en, fd; rtol=1e-3)

    println()
end


@testset "weighted project_doppler_comp!()" begin

    flux_star = rand(100, 20)
    weights = sqrt.(flux_star)
    μ = SSOF.make_template(flux_star, weights; min=0, max=1.2, use_mean=true)
    doppler_comp = SSOF.doppler_component(LinRange(5000,6000,100), μ)
    M = zeros(100, 3)
    s = zeros(20)

    data_tmp = copy(flux_star)
    data_tmp .-= μ
    rvs1 = SSOF.project_doppler_comp!(M, s, data_tmp, doppler_comp, ones(size(data_tmp)))
    s1 = copy(s)
    M1 = copy(M[:, 1])

    data_tmp = copy(flux_star)
    data_tmp .-= μ
    rvs2 = SSOF.project_doppler_comp!(M, s, data_tmp, doppler_comp, ones(size(data_tmp)))
    s2 = copy(s)
    M2 = copy(M[:, 1])

    data_tmp = copy(flux_star)
    data_tmp .-= μ
    rvs3 = SSOF.project_doppler_comp!(M, s, data_tmp, doppler_comp, weights)
    s3 = copy(s)
    M3 = copy(M[:, 1])

    @test M1 == M2 == M3
    @test rvs1 == rvs2
    @test s1 == s2
    @test rvs2 != rvs3
    @test s2 != s3

    println()
end

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


@testset "doppler_component_log_AD agrees with doppler_component_log" begin
    # Verifies the fix to the infinite-recursion bug (was: called itself instead of
    # doppler_component_AD). The AD variant and its non-AD sibling (doppler_component_log)
    # both compute doppler_component(λ, flux) ./ flux using the same finite-difference
    # scheme, so they should agree to machine precision.
    λ = collect(LinRange(5000.0, 6000.0, 30))
    flux = 1.0 .+ 0.2 .* sin.(λ ./ 500)

    result   = SSOF.doppler_component_log_AD(λ, flux)
    expected = SSOF.doppler_component_log(λ, flux)
    @test isapprox(result, expected; rtol=1e-12)

    println()
end

# Shared tiny Wobble dataset used by Tasks 3a, 3b, 3c
function _make_tiny_wobble_data(n_obs, n_epochs)
    log_λ_obs = collect(LinRange(8.78535, 8.78602, n_obs)) .+ 1e-5 .* (0:n_epochs-1)'
    log_λ_star = log_λ_obs
    flux = ones(n_obs, n_epochs) .+ 0.1 .* randn(n_obs, n_epochs)
    flux = max.(flux, 1e-6)
    var = fill(1e-4, n_obs, n_epochs)
    return SSOF.GenericData(flux, var, var, log_λ_obs, log_λ_star)
end

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

@testset "FlatLoss: Enzyme gradient via typed nested parameters" begin
    # Verifies that EnzymeFlatLossCache correctly differentiates loss(nested)
    # for both the multi-array case (time-variable scores, Vector{Array{Float64}}) and
    # the plain-vector case (RV optimization, Vector{Float64}).

    # Multi-array case: mimics [tel_s, star_s, rvs] from the time-variable finalize_scores! path
    tel_s  = rand(2, 5)
    star_s = rand(1, 5)
    rvs    = rand(5)
    pars_nested = [tel_s, star_s, rvs]
    flat_nested, unfl_nested = SSOF.flatten(pars_nested)
    loss_nested(p) = sum(abs2, p[1]) + 2 * sum(abs2, p[2]) + 3 * sum(abs2, p[3])

    f_nested = SSOF.FlatLoss(loss_nested, unfl_nested)
    cache_nested = SSOF.prepare_gradient(SSOF.EnzymeBackend(), f_nested, copy(flat_nested))
    val_n, ∂θ_n = SSOF.value_and_gradient!(cache_nested, f_nested, copy(flat_nested))

    fd_nested = est_∇(θ -> loss_nested(unfl_nested(θ)), copy(flat_nested))
    @test isfinite(val_n)
    @test isapprox(∂θ_n, fd_nested; rtol=1e-5)

    # Plain-vector case: mimics om.rv from the Wobble non-time-variable finalize_scores! path
    rv_vec = rand(8)
    flat_rv, unfl_rv = SSOF.flatten(rv_vec)
    loss_rv(v) = sum(abs2, v)

    f_rv = SSOF.FlatLoss(loss_rv, unfl_rv)
    cache_rv = SSOF.prepare_gradient(SSOF.EnzymeBackend(), f_rv, copy(flat_rv))
    val_r, ∂θ_r = SSOF.value_and_gradient!(cache_rv, f_rv, copy(flat_rv))

    @test isfinite(val_r)
    @test isapprox(∂θ_r, 2 .* rv_vec; rtol=1e-10)

    println()
end
