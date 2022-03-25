StellarSpectraObservationFitting/NEID
========

This folder has scripts which analyze NEID data with SSOF
- init.jl: Reformats the data into an SSOF-acceptable form
- param.jl: used by init.jl to filter which data to use. Hopefully will be deprecated soon.
- analysis.jl : Performs the model fitting
- gather.jl: Collects the results of many orders into single files
- reduce.jl: Combines results from different orders
