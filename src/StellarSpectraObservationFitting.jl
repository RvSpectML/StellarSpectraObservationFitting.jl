module StellarSpectraObservationFitting

include("general_functions.jl")
include("model_functions.jl")
include("EMPCA.jl")
include("DPCA_functions.jl")
include("flatten.jl")
include("ad_backend.jl")
include("optimization_functions.jl")
include("regularization_functions.jl")
include("continuum_functions.jl")
include("model_selection_functions.jl")
include("mask_functions.jl")
include("prior_gp_functions.jl")
include("rassine.jl")
include("error_estimation.jl")

# Precompile workload for PackageCompiler.create_sysimage.
# Covers all four data/model combinations so every Enzyme specialisation
# (loss closure type × AD cache type) is compiled into the sysimage.
# During ordinary Pkg.precompile() this block runs but Enzyme's LLVM-level
# compilation is NOT retained in the .ji pkgimage — only a sysimage captures
# native code.  Expect ~2–4 min per path when building a sysimage.
using PrecompileTools
@compile_workload begin
	n_obs, n_λ = 8, 30
	log_λ_vec  = range(log(5000.0), log(6000.0), length=n_λ) |> collect
	log_λ_obs  = repeat(log_λ_vec, 1, n_obs)
	log_λ_star = log_λ_obs .- 1e-5
	flux       = ones(n_λ, n_obs)
	var_       = ones(n_λ, n_obs)

	# GenericData (bounds computed internally)
	d_gen = GenericData(flux, var_, var_, log_λ_obs, log_λ_star)

	# LSFData: one square (n_λ × n_λ) identity LSF per observation
	lsf    = [SparseArrays.spdiagm(ones(n_λ)) for _ in 1:n_obs]
	d_lsf  = LSFData(flux, var_, var_, log_λ_obs, log_λ_star, lsf)

	for d in (d_gen, d_lsf)
		for dpca in (false, true)
			om = OrderModel(d; n_comp_tel=1, n_comp_star=1, dpca=dpca)

			# Adam path (TotalWorkspace handles both OrderModelWobble and OrderModelDPCA)
			mws_adam = TotalWorkspace(om, d)
			train_OrderModel!(mws_adam; iter=2, verbose=false)

			# L-BFGS path
			mws_bfgs = OptimTotalWorkspace(deepcopy(om), d)
			train_OrderModel!(mws_bfgs; iter=2, verbose=false)
		end
	end
end

end # module
