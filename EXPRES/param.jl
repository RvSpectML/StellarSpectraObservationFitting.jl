#global tophap_ccf_mask_scale_factor=1.6

global max_spectra_to_use = 250
global fits_target_str

global df_files
global df_files_use = df_files |>
   @filter( _.target == fits_target_str ) |>
   @take(max_spectra_to_use) |>
   DataFrame

df_files_use = df_files_use |>
   @orderby(_.bjd) |>
   @take(max_spectra_to_use) |>
   DataFrame
   
println("# Found ", size(df_files_use,1), " files of ",  size(df_files,1), " to process.")
