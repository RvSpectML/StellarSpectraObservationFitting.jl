@testset "rv_to_D / D_to_rv roundtrip" begin
    for v in [-30_000.0, 0.0, 15_000.0, 3e5]
        @test isapprox(SSOF.D_to_rv(SSOF.rv_to_D(v)), v; rtol=1e-8)
    end
    @test SSOF.rv_to_D(0.0) == 0.0
    println()
end

@testset "searchsortednearest" begin
    a = [1.0, 2.0, 3.0, 5.0]
    @test SSOF.searchsortednearest(a, 1.0) == 1   # exact match at start
    @test SSOF.searchsortednearest(a, 0.5) == 1   # below first element
    @test SSOF.searchsortednearest(a, 6.0) == 4   # above last element
    @test SSOF.searchsortednearest(a, 2.4) == 2   # closer to 2.0
    @test SSOF.searchsortednearest(a, 2.6) == 3   # closer to 3.0
    @test SSOF.searchsortednearest(a, 2.5; lower=true) == 2  # lower flag returns lower index

    # vector query must match element-wise scalar results
    xs = [1.1, 2.9, 4.9]
    inds = SSOF.searchsortednearest(a, xs)
    @test inds == [SSOF.searchsortednearest(a, x) for x in xs]
    println()
end

@testset "weighted_mean and make_template" begin
    x = [1.0 3.0 5.0; 2.0 4.0 6.0]
    σ² = ones(2, 3)
    # uniform weights → weighted mean equals arithmetic mean
    wm = SSOF.weighted_mean(x, σ²; dims=2)
    @test isapprox(vec(wm), [mean(x[1, :]), mean(x[2, :])]; rtol=1e-10)

    # all-Inf variance in one column → default value 0 used for that column's contribution
    σ²_bad = ones(2, 3)
    σ²_bad[:, 2] .= Inf
    wm_bad = SSOF.weighted_mean(x, σ²_bad; dims=2)
    @test all(isfinite, wm_bad)

    # make_template unweighted: median across columns
    M = [1.0 2.0 10.0; 3.0 4.0 5.0]
    t = SSOF.make_template(M)
    @test isapprox(t, [median(M[1, :]), median(M[2, :])]; rtol=1e-10)

    # make_template mean form
    t_mean = SSOF.make_template(M; use_mean=true)
    @test isapprox(t_mean, vec(mean(M; dims=2)); rtol=1e-10)
    println()
end

@testset "observation_night_inds single night" begin
    times = [0.1, 0.2, 0.3]
    ni = SSOF.observation_night_inds(times)
    @test length(ni) == 1
    @test ni[1] == 1:3
    println()
end
