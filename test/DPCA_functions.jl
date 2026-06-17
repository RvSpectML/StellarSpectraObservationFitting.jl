@testset "simple_derivative vs simple_derivative_AD" begin
    # Both implement the same FD stencil; results must be exactly equal
    x = collect(1.0:10.0)
    @test SSOF.simple_derivative(x) == SSOF.simple_derivative_AD(x)

    # Interior points use central differences; endpoints use forward/backward
    y = collect(Float64, 1:5)
    dy = SSOF.simple_derivative(y)
    @test dy[1] == y[2] - y[1]          # forward difference at left endpoint
    @test dy[end] == y[end] - y[end-1]  # backward difference at right endpoint
    @test dy[3] == (y[4] - y[2]) / 2    # central difference at interior point
    println()
end

@testset "doppler_component vs doppler_component_AD" begin
    λ = collect(LinRange(5000.0, 6000.0, 30))
    flux = 1.0 .+ 0.2 .* sin.(λ ./ 500)
    @test isapprox(
        SSOF.doppler_component(λ, flux),
        SSOF.doppler_component_AD(λ, collect(flux));
        rtol=1e-10)
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

@testset "DEMPCA! smoke test" begin
    n, n_epochs = 12, 5
    # M[:,1] = Doppler basis (save_doppler_in_M1=true default), M[:,2] = extra component.
    # scores must have the same number of rows as M has columns.
    n_comp = 2
    M = zeros(n, n_comp)
    scores = zeros(n_comp, n_epochs)  # row 1 = rv_scores, row 2 = extra component
    μ = ones(n) .+ 0.05 .* randn(n)
    data_temp = ones(n, n_epochs) .+ 0.01 .* randn(n, n_epochs)
    weights = ones(n, n_epochs)
    doppler_comp = SSOF.doppler_component(collect(LinRange(5000.0, 6000.0, n)), μ)

    rvs_out = SSOF.DEMPCA!(
        M, scores, view(scores, 1, :), copy(μ),
        copy(data_temp), copy(weights), doppler_comp;
        use_log=false, niter=3)

    @test length(rvs_out) == n_epochs
    @test all(isfinite, rvs_out)
    println()
end
