#global tophap_ccf_mask_scale_factor=1.6

global max_spectra_to_use = 150
global fits_target_str

if fits_target_str == "Solar" || fits_target_str == "Sun"
	global df_files

	# is the right star
	global df_files_use = df_files |>
		@filter( _.target == fits_target_str ) |>
		DataFrame

	# same good calibration available
	df_files_use = df_files_use |>
		@filter( _.driftfun == "dailymodel0" ) |>
		DataFrame

	# good weather
	pyrheliometer_fn = joinpath(neid_pyr_data_path * target_subdir, "pyrheliometer.csv")
	df_pyrohelio_obs = CSV.read(pyrheliometer_fn, DataFrame)
	df_pyrohelio_obs.filename = replace.(df_pyrohelio_obs.filename,"neidL0_"=>"neidL2_")
	@assert size(df_pyrohelio_obs,1) >= 1
	df_files_use.filename = basename.(df_files_use.Filename)
	df_files_use = leftjoin(df_files_use, df_pyrohelio_obs, on=:filename, makeunique=true)
	df_files_use = df_files_use |>
		@filter( _.rms_pyroflux <= 0.0035* _.mean_pyroflux ) |>
		DataFrame

	# no obstructions or absolutely terrible pointing
	df_files_use = df_files_use |>
		@filter( _.expmeter_mean >= 6e4 ) |>
		DataFrame

	# low airmass
	df_files_use = df_files_use |>
		@filter( _.airmass <= 2 || _.airmass > 13 ) |>
		DataFrame


	# within 2 hours of solar noon
	using PyCall
	AstropyCoordinates = PyNULL()
	AstropyTime = PyNULL()

	copy!(AstropyCoordinates , pyimport("astropy.coordinates") )
	copy!(AstropyTime , pyimport("astropy.time") )

	function get_WIYN_solar_hour_angle(jd::Vector{<:Real})

		# getting local sidereal time
		loc = Dict("lon"=> -111.600562, "lat"=> 31.958092, "elevation"=> 2.091)  # wiyn location
		pyloc = AstropyCoordinates.EarthLocation.from_geodetic(loc["lon"], loc["lat"], height=loc["elevation"])
		TimeObs = AstropyTime.Time(jd, format="jd", scale="utc", location=pyloc)
		lst = TimeObs.sidereal_time("mean")  # doesn't work on ROAR :(

		# getting sun right ascention
		ra = AstropyCoordinates.get_sun(TimeObs).ra

		# converting ra to hour angle
		ha = lst .- ra./15
		ha[ha.<-12] .+= 24
		ha[ha.>12] .-= 24
		return ha

	end

	df_files_use[!, :solar_hour_angle] = abs.(get_WIYN_solar_hour_angle(df_files_use.bjd))
		df_files_use = df_files_use |>
		@filter( _.solar_hour_angle <= 2 ) |>
		DataFrame

	# sort and take the max amount
	df_files_use = df_files_use |>
	  	@orderby(_.bjd) |>
	  	@take(max_spectra_to_use) |>
	  	DataFrame
else
	global df_files
	global df_files_use = df_files |>
	   @filter( _.target == fits_target_str ) |>
	   @take(max_spectra_to_use) |>
	   DataFrame
end

println("# Found ", size(df_files_use,1), " files of ",  size(df_files,1), " to process.")
