## Julia version 1.10
using Pkg
#Pkg.generate("BLPSFU")
cd("BLPSFU") # Change directory to your project folder
Pkg.activate(".")
Pkg.instantiate()

#= 
Estimation of demand parameters through the BLP method.

The BLP objective function and its gradient are defined in BLP_functions and BLP_gradient, respectively.
Documentation of the BLP method and these functions is in the corresponding modules.

Uses Optim optimization package to find the θ₂ that minimizes the function.
The estimate for θ₂ is used to recover estimates of θ₁ (elasticities of mean demand) from the objective function.
=#

# Load key functions and packages -------------------------------------------------

cd("/Users/victoraguiar/Documents/GitHub/Julia-BLP/code")

include("demand_functions.jl")    # module with custom BLP functions (objective function and σ()/shares)
include("demand_instruments.jl")  # module to calculate BLP instruments
include("demand_derivatives.jl")  # module with gradient function 

using .demand_functions
using .demand_instrument_module
using .demand_derivatives

using CSV               # loading data
using DataFrames        # loading data
using LinearAlgebra     # basic math
using Statistics        # for mean


# Load key data ------------------------------------------------------------------
cd("/Users/victoraguiar/Documents/GitHub/Julia-BLP/data and random draws")

blp_data = CSV.read("BLP_product_data.csv", DataFrame) # dataframe with all observables 
v_50 = Matrix(CSV.read("random_draws_50_individuals.csv", DataFrame, header=0)) # pre-selected random draws from joint normal to simulate 50 individuals
# reshape to 3-d arrays: v(market, individual, coefficient draw) 
v_50 = reshape(v_50, (20,50,5)) # 20 markets, 50 individuals per market, 5 draws per invididual (one for each θ₂ random effect coefficient)

# Load X variables. 2217x5 and 2217x6 matrices respectively
X = Matrix(blp_data[!, ["price","const","hpwt","air","mpg","space"]]) # exogenous x variables and price
# Load Y variable market share. 2217x1 vector
share = Vector(blp_data[!,"share"])
# product, market, and firm ids 
id = Vector(blp_data[!,"id"])
cdid = Vector(blp_data[!,"cdid"])
firmid = Vector(blp_data[!,"firmid"])

# BLP instruments. Following BLP95.
# price (column 1) not included in BLP instruments.

#= BLP instruments =#
Z = BLP_instruments(X[:,Not(1)], id, cdid, firmid)


# Minimize objective function -----------------------------------------------------
using Optim             # for minimization functions
using BenchmarkTools    # for timing/benchmarking functions

# θ₂ guess values. Initialze elements as floats.
θ₂ = [0.0, 0.0, 0.0, 0.0, 0.0] # this implies starting θ₁ values equal to the IV coefficients (random effects = 0)
θ₂ = [1.0, 1.0, 1.0, 1.0, 1.0] # alternative starting point from which all algorithms converge

# test run and timing of objective function and gradient
# Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z,v_50,cdid) # returns 4 values  
@btime demand_objective_function($θ₂,$X,$share,$Z,$v_50,$cdid)  
# Usually <100ms. Speed varies depending on θ₂.

# g = gradient(θ₂,X,Z,v_50,cdid,ξ,𝒯)
# @btime gradient($θ₂,$X,$Z,$v_50,$cdid,$ξ,$𝒯)
# ~ 1.1 seconds. 


# temporary ananomyous functions for objective function and gradient
function f(θ₂)
    # run objective function and get key outputs
    Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z,v_50,cdid)
    # return objective function value
    return Q
end

function ∇(storage, θ₂)
    # run objective function to update ξ and 𝒯 values for new θ₂
    Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z,v_50,cdid)
    # calculate gradient and record value
    g = gradient(θ₂,X,Z,v_50,cdid,ξ,𝒯)
    storage[1] = g[1]
    storage[2] = g[2]
    storage[3] = g[3]
    storage[4] = g[4]
    storage[5] = g[5]
end

# optimization routines
result = optimize(f, θ₂, NelderMead(), Optim.Options(x_tol=1e-3, iterations=500, show_trace=true, show_every=10))
result = optimize(f, ∇, θ₂, LBFGS(), Optim.Options(x_tol=1e-2, iterations=50, show_trace=true, show_every=1))   
result = optimize(f, ∇, θ₂, BFGS(), Optim.Options(x_tol=1e-2, iterations=50, show_trace=true, show_every=1))
result = optimize(f, ∇, θ₂, GradientDescent(), Optim.Options(x_tol=1e-2, iterations=50, show_trace=true, show_every=1))
result = optimize(f, ∇, θ₂, ConjugateGradient(), Optim.Options(x_tol=1e-2, iterations=50, show_trace=true, show_every=1))

# get results 
θ₂ = Optim.minimizer(result)
θ₁ = demand_objective_function(θ₂,X,share,Z,v_50,cdid)[2]

# approximate solution
# θ₂ = [ 0.172, -2.528, 0.763, 0.589,  0.595]
# θ₁ = [-0.427, -9.999, 2.801, 1.099, -0.430, 2.795]
