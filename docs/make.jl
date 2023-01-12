using Documenter
using StellarSpectraObservationFitting

makedocs(
    sitename = "StellarSpectraObservationFitting",
    format = Documenter.HTML(),
    modules = [StellarSpectraObservationFitting]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
