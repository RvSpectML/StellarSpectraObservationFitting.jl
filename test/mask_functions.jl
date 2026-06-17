@testset "insert_and_dedup!" begin
    v = [1, 3, 5]
    SSOF.insert_and_dedup!(v, 3)   # duplicate — no change
    @test v == [1, 3, 5]
    SSOF.insert_and_dedup!(v, 4)   # new element inserted in order
    @test v == [1, 3, 4, 5]
    println()
end

@testset "affected_pixels" begin
    bad = [false, true, false, true, false]
    aff = SSOF.affected_pixels(bad)
    @test aff == [2, 4]

    # matrix: collects unique first-dimension indices
    bad_mat = [false true; true false; false true]  # rows 1,2,3 involved
    aff_mat = SSOF.affected_pixels(bad_mat)
    @test aff_mat == [1, 2, 3]
    println()
end

@testset "mask! vector form (boolean mask)" begin
    # The AbstractVecOrMat dispatch takes a boolean bad-pixel mask
    var = [1.0, 2.0, 3.0, 4.0]
    bad = [false, true, false, true]
    SSOF.mask!(var, bad)
    @test var[2] == Inf
    @test var[4] == Inf
    @test var[1] == 1.0
    @test var[3] == 3.0
    println()
end

@testset "mask! matrix form (no padding)" begin
    var = ones(5, 3)
    SSOF.mask!(var, [2, 4])
    @test all(var[2, :] .== Inf)
    @test all(var[4, :] .== Inf)
    @test all(var[1, :] .== 1.0)
    @test all(var[3, :] .== 1.0)
    println()
end

@testset "mask! matrix form with padding" begin
    var = ones(10, 3)
    SSOF.mask!(var, [5]; padding=1)
    # rows 4, 5, 6 should be masked
    @test all(var[4, :] .== Inf)
    @test all(var[5, :] .== Inf)
    @test all(var[6, :] .== Inf)
    @test all(var[3, :] .== 1.0)
    @test all(var[7, :] .== 1.0)
    println()
end
