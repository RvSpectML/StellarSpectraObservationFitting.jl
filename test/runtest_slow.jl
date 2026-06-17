# Standalone slow tests — run with: include("test/runtest_slow.jl")
# Tests _eval_lm for every plausible combination of n_comp_tel ∈ {0,1,2}
# and n_comp_star ∈ {0,1,2}, plus the BaseLinearModel (no free mean) case.

using Test
using LinearAlgebra
import StellarSpectraObservationFitting as SSOF

function _make_data(n_obs=20, n_epochs=4)
    log_λ_obs = collect(LinRange(8.78535, 8.78602, n_obs)) .+ 1e-5 .* (0:n_epochs-1)'
    flux = ones(n_obs, n_epochs) .+ 0.1 .* randn(n_obs, n_epochs)
    flux = max.(flux, 1e-6)
    var = fill(1e-4, n_obs, n_epochs)
    return SSOF.GenericData(flux, var, var, log_λ_obs, log_λ_obs)
end

# Reference implementations of _eval_lm formulas for test verification
_ref_eval(lm::SSOF.TemplateModel)    = lm.μ * ones(lm.n)'
_ref_eval(lm::SSOF.FullLinearModel)  = lm.log ? exp.(lm.M * lm.s) .* lm.μ : lm.M * lm.s .+ lm.μ
_ref_eval(lm::SSOF.BaseLinearModel)  = lm.log ? exp.(lm.M * lm.s) : lm.M * lm.s

@testset "_eval_lm: all (n_tel, n_star) combinations via OrderModel" begin
    # OrderModel uses log_lm=true by default (_log_lm_default), so n_comp>0 → FullLinearModel(log=true)
    # and n_comp=0 → TemplateModel (always linear-space, returns μ broadcast across epochs).
    d = _make_data()
    n_epochs = size(d.flux, 2)

    for n_tel in 0:2, n_star in 0:2
        @testset "n_tel=$n_tel n_star=$n_star" begin
            om = SSOF.OrderModel(d; n_comp_tel=n_tel, n_comp_star=n_star, oversamp=false)
            n_tel_pix  = length(om.tel.log_λ)
            n_star_pix = length(om.star.log_λ)

            tel_out  = SSOF._eval_lm(om.tel.lm)
            star_out = SSOF._eval_lm(om.star.lm)

            @test size(tel_out)  == (n_tel_pix, n_epochs)
            @test size(star_out) == (n_star_pix, n_epochs)
            @test all(isfinite, tel_out)
            @test all(isfinite, star_out)
            @test isapprox(tel_out,  _ref_eval(om.tel.lm))
            @test isapprox(star_out, _ref_eval(om.star.lm))

            if n_tel == 0
                @test om.tel.lm  isa SSOF.TemplateModel
            else
                @test om.tel.lm  isa SSOF.FullLinearModel
            end
            if n_star == 0
                @test om.star.lm isa SSOF.TemplateModel
            else
                @test om.star.lm isa SSOF.FullLinearModel
            end

            println("n_tel=$n_tel n_star=$n_star  tel=$(size(tel_out)) star=$(size(star_out))")
        end
    end
end

@testset "_eval_lm: FullLinearModel log=false (linear space)" begin
    # OrderModel defaults to log_lm=true; test the linear (log=false) branch directly.
    n_pix, n_epochs, n_comp = 30, 5, 2
    M = 0.1 .* randn(n_pix, n_comp)
    s = 0.05 .* randn(n_comp, n_epochs)
    μ = ones(n_pix) .+ 0.05 .* randn(n_pix)

    flm = SSOF.FullLinearModel(M, s, μ, false)   # log=false
    out = SSOF._eval_lm(flm)
    @test size(out) == (n_pix, n_epochs)
    @test all(isfinite, out)
    @test isapprox(out, _ref_eval(flm))
    println("FullLinearModel log=false: size=$(size(out))")
end

@testset "_eval_lm: BaseLinearModel (no free telluric mean, μ implicit ones)" begin
    # BaseLinearModel arises when include_mean=false in Submodel (not reachable via OrderModel
    # kwargs directly). The telluric template is effectively pinned to ones, with feature
    # vectors absorbing all telluric variation. Tests both log and linear-space variants.
    n_pix, n_epochs, n_comp = 30, 5, 2
    M = 0.1 .* randn(n_pix, n_comp)
    s = 0.05 .* randn(n_comp, n_epochs)

    for use_log in (true, false)
        @testset "log=$use_log" begin
            blm = SSOF.BaseLinearModel(M, s, use_log)
            out = SSOF._eval_lm(blm)
            @test size(out) == (n_pix, n_epochs)
            @test all(isfinite, out)
            @test isapprox(out, _ref_eval(blm))
            println("BaseLinearModel log=$use_log: size=$(size(out))")
        end
    end
end
