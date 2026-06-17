if !SKIP_SLOW; @testset "fit_regularization! smoke test" begin
    d = _make_tiny_wobble_data(30, 4)
    om = SSOF.OrderModel(d; n_comp_tel=1, n_comp_star=1, oversamp=false)
    mws = SSOF.TotalWorkspace(om, d)
    SSOF.fit_regularization!(mws; verbose=false)
    # All non-zero regularization values must be finite and positive
    for (_, v) in mws.om.reg_tel
        if v != 0; @test isfinite(v) && v > 0 end
    end
    for (_, v) in mws.om.reg_star
        if v != 0; @test isfinite(v) && v > 0 end
    end
    println()
end; end  # if !SKIP_SLOW
