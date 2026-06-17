@testset "flatten LinearModel roundtrip" begin
    M = rand(4, 2)
    s = rand(2, 5)
    μ = rand(4)
    lm = SSOF.FullLinearModel(M, s, μ, false)

    p0, unflatten = SSOF.flatten(lm)
    lm2 = unflatten(p0)
    @test isapprox(lm2.M, M; rtol=1e-10)
    @test isapprox(lm2.s, s; rtol=1e-10)
    @test isapprox(lm2.μ, μ; rtol=1e-10)

    # perturbing the flat vector produces a different model
    p1 = copy(p0)
    p1[1] += 1.0
    lm3 = unflatten(p1)
    @test !isapprox(vec(lm3.M), vec(lm.M); rtol=1e-10)
    println()
end

@testset "flatten nested Vector{Array} roundtrip" begin
    pars = [rand(3, 2), rand(2, 3), rand(5)]
    p0, unflatten = SSOF.flatten(pars)
    pars2 = unflatten(p0)
    for i in eachindex(pars)
        @test isapprox(pars2[i], pars[i]; rtol=1e-10)
    end
    @test length(p0) == sum(length.(pars))
    println()
end
