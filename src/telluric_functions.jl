using FITSIO
using DataInterpolations

teldir = "/home/kment/RV/telluric"      # Directory containing telluric model spectra

get_h2o_spectrum(λ_values) = interp_telluric_file("$teldir/kpno_telluric_4.0_0.0_h2o_only.fits", λ_values)
get_oxy_spectrum(λ_values) = interp_telluric_file("$teldir/kpno_telluric_4.0_0.0_oxy_only.fits", λ_values)
get_tel_spectrum(λ_values) = interp_telluric_file("$teldir/kpno_telluric_4.0_0.0.fits", λ_values)

"""
    get_telluric_model(λ_values, pwv, z_angle; verbose=true)

Estimate the telluric spectrum at a range of wavelenths (λ_values) for a specific zenith angle
(z_angle, in degrees) and precipitable water vapor (pwv) value.
"""
function get_telluric_model(λ_values, pwv, z_angle; verbose::Bool=true)

    @assert pwv <= 16 "Precipitable water vapor cannot exceed 16."
    @assert z_angle < 60 "Zenith angle must be below 60."

    pwv_files = [1, 2, 4, 8, 16]
    z_files = [0, 34, 44, 51, 60]
    i_pwv = 1   # Index of PWV file
    i_z = 1     # Index of zenith angle file

    for (i, x) in enumerate(pwv_files)
        if pwv > x
            i_pwv = i + 1
        end
    end
    for (i, x) in enumerate(z_files)
        if z_angle >= x
            i_z = i
        end
    end

    fn_model1 = "$teldir/kpno_telluric_$(pwv_files[i_pwv-1]).0_$(z_files[i_z]).0.fits"
    fn_model2 = "$teldir/kpno_telluric_$(pwv_files[i_pwv]).0_$(z_files[i_z]).0.fits"
    fn_model3 = "$teldir/kpno_telluric_$(pwv_files[i_pwv-1]).0_$(z_files[i_z+1]).0.fits"
    fn_model4 = "$teldir/kpno_telluric_$(pwv_files[i_pwv]).0_$(z_files[i_z+1]).0.fits"
    if verbose
        println("Using models: $fn_model1 $fn_model2 $fn_model3 $fn_model4")
    end

    t_model1 = interp_telluric_file(fn_model1, λ_values)
    t_model2 = interp_telluric_file(fn_model2, λ_values)
    t_model3 = interp_telluric_file(fn_model3, λ_values)
    t_model4 = interp_telluric_file(fn_model4, λ_values)

    X = sec(z_angle * π / 180)          # Airmass
    X_model12 = sec(z_files[i_z] * π / 180)
    X_model34 = sec(z_files[i_z+1] * π / 180)
    X_power12 = X / X_model12
    X_power34 = X / X_model34
    pwv_power12 = (pwv - pwv_files[i_pwv-1]) / (pwv_files[i_pwv] - pwv_files[i_pwv-1]) * X_power12
    t_scaled12 = t_model1 .^ (X_power12 - pwv_power12) .* t_model2 .^ pwv_power12
    pwv_power34 = (pwv - pwv_files[i_pwv-1]) / (pwv_files[i_pwv] - pwv_files[i_pwv-1]) * X_power34
    t_scaled34 = t_model3 .^ (X_power34 - pwv_power34) .* t_model4 .^ pwv_power34

    weight = (X - X_model12) / (X_model34 - X_model12)
    t_scaled = t_scaled12 .* weight .+ t_scaled34 .* (1 - weight)

    return t_scaled

end

"""
    get_telluric_prior_spectrum(λ_values, species::Symbol=:all)

Evaluate the spectrum used as a telluric prior at a range of wavelenths (λ_values).
"""
function get_telluric_prior_spectrum(λ_values, species::Symbol=:all)

    if species == :all
        return get_tel_spectrum(λ_values)
    elseif species == :h2o
        return get_h2o_spectrum(λ_values)
    elseif species == :oxy
        return get_oxy_spectrum(λ_values)
    end

end

"""
    interp_telluric_file(filename, λ_values)

Estimate the telluric spectrum at a range of wavelenths (λ_values) by interpolating a model spectrum
in a given FITS file.
"""
function interp_telluric_file(filename, λ_values)

    f = FITS(filename)
    λ_all = reverse(read(f[2], "wavelength")) .* 10
    t_all = reverse(read(f[2], "transmittance"))

    interp = LinearInterpolation(t_all, λ_all)
    t_model = interp.(λ_values)
    t_model[t_model .< 0] .= 0

    return t_model

end

"""
    simulate_telluric_data(λ_values, pwv_values, z_values)

Simulate the telluric spectrum at a range of wavelenths (λ_values) for a list of precipitable water
vapor values (pwv_values) and zenith angle values (z_values, in degrees).
"""
function simulate_telluric_data(λ_values, pwv_values, z_values)

    @assert length(pwv_values) == length(z_values) "Must provide an equal number of PWV and zenith angle values."

    t_all = Matrix{Float64}(undef, length(λ_values), length(pwv_values))

    for i in range(1, length(pwv_values))
        t_all[:,i] = get_telluric_model(λ_values, pwv_values[i], z_values[i]; verbose=false)
    end

    return t_all

end