@testset "AD backends load" begin
    @test SSOF.MooncakeBackend() isa SSOF.ADBackend
    @test SSOF.EnzymeBackend() isa SSOF.ADBackend
end

if !SKIP_SLOW; @testset "MooncakeBackend gradient correctness" begin
    f(x) = sum(x .^ 2)
    x0 = randn(8)
    cache = SSOF.prepare_gradient(SSOF.MooncakeBackend(), f, copy(x0))
    val_mk, ∂_mk = SSOF.value_and_gradient!(cache, f, copy(x0))
    @test isapprox(val_mk, sum(x0 .^ 2); rtol=1e-10)
    @test isapprox(∂_mk, 2 .* x0; rtol=1e-4)
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
