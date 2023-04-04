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
        "Getting started" => "gettingstarted.md",
        "User's guide" => [
            "Data preparation" => "data.md",
            # "Creating a SSOF model" => [
            #     "Common structures" => "structs.md",
            #     "Initialization and model selection" => "init.md"
            #     "Optimization" => "opt.md",
            #     ],
            # "Regularization" => [
            #     "Regularization functions" => "prior.md",
            #     "𝒪(n) GP Regularization" => "prior_gp.md"
            #     ],
            # "Model error estimation" => "error.md",
        ],
        "Various other functions" => [
            # "(D)EMPCA" => "empca.md",
            # "Utility functions" => "general.md",
            # "Model functions" => "model.md",
            # "Rassine-like continuum fitting" => "continuum.md",
            "Everything else" => "indices.md",
        ],
        "LICENSE.md",
    ]
)

deploydocs(
    repo = "github.com/christiangil/StellarSpectraObservationFitting.jl.git",
    deploy_config = Documenter.GitHubActions(),
)