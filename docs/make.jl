using Documenter
using StellarSpectraObservationFitting

# DocMeta.setdocmeta!(StellarSpectraObservationFitting, :DocTestSetup, :(using StellarSpectraObservationFitting); recursive=true)

makedocs(
    sitename = "StellarSpectraObservationFitting.jl",
    format = Documenter.HTML(),
    modules = [StellarSpectraObservationFitting],
    authors = "Christian Gilbertson",
    pages = [
        "Home" => "index.md",
        hide("Indices" => "indices.md"),
        "LICENSE.md",
    ]
)

deploydocs(
    repo = "github.com/christiangil/StellarSpectraObservationFitting.jl.git",
    deploy_config = Documenter.GitHubActions(),
)