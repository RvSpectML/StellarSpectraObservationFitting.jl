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
        "gettingstarted.md",
        # "User's Guide" => [
        #     "Kernels" => [
        #         "Kernel functions" => "kernel.md",
        #         "Adding new kernels" => "kernel_creation.md",
        #         ],
        #     "GPLinearODE struct" => "glo.md",
        #     "GLOM Functionality" => "nlogl.md",
        #     "Prior functions" => "priors.md",
        # ],
        hide("(D)EMPCA" => "empca.md"),
        hide("Continuum" => "continuum.md"),
        hide("Indices" => "indices.md"),
        # hide("Diagnostic functions" => "diagnostic.md"),
        # hide("Utility functions" => "utility.md"),
        "LICENSE.md",
    ]
)

deploydocs(
    repo = "github.com/christiangil/StellarSpectraObservationFitting.jl.git",
    deploy_config = Documenter.GitHubActions(),
)