#global tophap_ccf_mask_scale_factor=1.6

global max_spectra_to_use = 200
if max_spectra_to_use < 200
   @warn "param.in setting max_spectra_to_use to " * string(max_spectra_to_use)
end
global fits_target_str


if fits_target_str == "HD 10700"
   global espresso_mask_filename = "G8.espresso.mas"
   global ccf_mid_velocity = -16640.0
   global df_files
   global df_files_use = df_files |>
      @filter( _.target == fits_target_str ) |>
      @take(max_spectra_to_use) |>
      DataFrame
end
