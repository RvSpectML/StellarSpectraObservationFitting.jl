StellarSpectraObservationFitting/EXPRES
========

This folder has scripts which analyze EXPRES data with SSOF
- init.jl: Reformats the data into an SSOF-acceptable form
- param.jl: used by init.jl to filter which data to use. Hopefully will be deprecated soon.
- analysis.jl : Performs the model fitting
- gather.jl: Collects the results of many orders into single files
- reduce.jl: Combines results from different orders

Additionally,
- clean_dir.jl: Cleans old files from the save directories
- comp_testing.jl: An older script that investigated what RVs were output when a small, constant ammount of basis vectors were used
- gp_prior_performance.jl: Shows that my sparse GP prior code is fast and accurate
- lsf.jl: Defines a function that outputs the LSF width as a function of wavelength
- order_masks.jl: Shows how one might start to mask really bad behaving lines
- scratch.jl: Random, ugly code that I should delete
