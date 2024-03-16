## Julia version 1.10
#Pkg.generate("BLPSFU")
using Pkg

using ProjectRoot
using CSV               # loading data
using DataFrames        # loading data
using LinearAlgebra     # basic math
using Statistics        # for mean

#cd("BLPSFU") # Change directory to your project folder
Pkg.activate("BLPSFU")
Pkg.instantiate()

# Determine the current directory

rootdir = @projectroot()

#= 
Estimation of demand parameters through the BLP method.

The BLP objective function and its gradient are defined in BLP_functions and BLP_gradient, respectively.
Documentation of the BLP method and these functions is in the corresponding modules.

Uses Optim optimization package to find the θ₂ that minimizes the function.
The estimate for θ₂ is used to recover estimates of θ₁ from the objective function.
=#

# Load key functions and packages -------------------------------------------------

include(@projectroot("code", "demand_functions.jl"))    # module with custom BLP functions (objective function and σ())
include(@projectroot("code", "demand_instruments.jl")) # module to calculate BLP instruments
include(@projectroot("code", "demand_derivatives.jl")) # module with gradient function 

using .demand_functions
using .demand_instrument_module
using .demand_derivatives

# Load key data ------------------------------------------------------------------

blp_data_path = @projectroot("data and random draws", "BLP_product_data.csv")

random_draws_path = @projectroot("data and random draws", "random_draws_50_individuals.csv")

blp_data = CSV.read(blp_data_path, DataFrame) # dataframe with all observables 

v_50 = Matrix(CSV.read(random_draws_path, DataFrame, header=0)) # pre-selected random draws from joint normal to simulate 50 individuals
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

# BLP instruments. Function uses same code as Question 1b to calculate instruments.
# price (column 1) not included in BLP instruments.

#= BLP instruments =#
# function to enclose the calculation of instruments.
# same code as Demand Side - OLS and 2SLS, packaged as a function to save space.

#= Two sets of instruments
1. Characteristics of other products from the same company in the same market.
Logic: the characteristics of other products affect the price of a 
given product but not its demand. Alternatively, firms decide product characteristics X 
before observing demand shocks ξ. 
2. Characteristics of other products from different companies in the same market.
Logic: the characteristics of competing products affects the price of a
given product but not its demand. Alternatively, other firms decide their product
characteristics X without observing the demand shock for the given product ξ.
=#
function BLP_instruments(X, id, cdid, firmid)

    n_products = size(id,1) # number of observations = 2217

    # initialize arrays to hold the two sets of 5 instruments. 
    IV_others = zeros(n_products,5)
    IV_rivals = zeros(n_products,5)

    # loop through every product in every market (every observation)
    for j in 1:n_products
        # 1. Set of instruments from other product characteristics
        # get the index of all different products (id) made by the same firm (firmid)
        # in the same market/year (cdid) 
        other_index = (firmid.==firmid[j]) .* (cdid.==cdid[j]) .* (id.!=id[j])
        # x variable values for other products (excluding price)
        other_x_values = X[other_index,:]
        # sum along columns
        IV_others[j,:] = sum(other_x_values, dims=1)
        # 2. Set of instruments from rival product characteristics
        # get index of all products from different firms (firmid) in the same market/year (cdid)
        rival_index = (firmid.!=firmid[j]) .* (cdid.==cdid[j])
        # x variable values for other products (excluding price)
        rival_x_values = X[rival_index,:]
        # sum along columns
        IV_rivals[j,:] = sum(rival_x_values, dims=1)
    end
    # vector of observations and instruments
    IV = [X IV_others IV_rivals]
    return IV
end

Z = BLP_instruments(X[:,Not(1)], id, cdid, firmid)
# export instruments to csv

CSV.write("BLP_instruments.csv", DataFrame(Z, :auto))

# Exploring my demand objective function -----------------------------------------------------

#δ = zeros(size(share))

#n_individuals = size(v_50,2)
#n_products = size(X,1)

#θ₂ = [0.0, 0.0, 0.0, 0.0, 0.0]

# initial guess for θ₁. Random effects are set to 0.

# repeat δ for each individual. 2217x50 matrix

#δ = repeat(δ,1,n_individuals) 


#μ = zeros(n_products, n_individuals)

#for market in unique(cdid)
    #μ[cdid.==market,:] = X[cdid.==market,Not(3)] * (v_50[market,:,:] .* θ₂')' 
#end

#μ

# Minimize objective function -----------------------------------------------------
using Optim             # for minimization functions
using BenchmarkTools    # for timing/benchmarking functions

# θ₂ guess values. Initialze elements as floats.
θ₂ = [0.0, 0.0, 0.0, 0.0, 0.0] # this implies starting θ₁ values equal to the IV coefficients (random effects = 0)
θ₂ = [1.0, 1.0, 1.0, 1.0, 1.0] # alternative starting point from which all algorithms converge

# test run and timing of objective function and gradient
# Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z,v_50,cdid) # returns 4 values  
# @btime demand_objective_function($θ₂,$X,$share,$Z,$v_50,$cdid)  
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
# result = optimize(f, ∇, θ₂, LBFGS(), Optim.Options(x_tol=1e-2, iterations=50, show_trace=true, show_every=1))   
# result = optimize(f, ∇, θ₂, BFGS(), Optim.Options(x_tol=1e-2, iterations=50, show_trace=true, show_every=1))
# result = optimize(f, ∇, θ₂, GradientDescent(), Optim.Options(x_tol=1e-2, iterations=50, show_trace=true, show_every=1))
result = optimize(f, ∇, θ₂, ConjugateGradient(), Optim.Options(x_tol=1e-2, iterations=50, show_trace=true, show_every=1))

# get results 
θ₂ = Optim.minimizer(result)
θ₁ = demand_objective_function(θ₂,X,share,Z,v_50,cdid)[2]

# approximate solution
# θ₂ = [ 0.172, -2.528, 0.763, 0.589,  0.595]
# θ₁ = [-0.427, -9.999, 2.801, 1.099, -0.430, 2.795]

## Using JuMP and Ipopt
using JuMP
using Ipopt

## Objective Function
BLPdemand = JuMP.Model(Ipopt.Optimizer) 
JuMP.@variable(BLPdemand, θ[1:2])

#= Demand Objective Function -----------------------------------------------------------
Performs the key steps for BLP demand estimation

Key steps:
1. given θ₂, solve for δ using contraction mapping
2. using δ, calculate ξⱼ = δ - xⱼθ₁
3. set up GMM moments: E[Z*ξ(θ)] = G(θ) = 0 and construct GMM function Q(θ) = G(θ)'*W*G(θ)
4. return GMM function value

Step 1. requires calculating predicted market share given δ. 
This is done with the second function here, σ().

Inputs:

θ₂: 5x1 vector of σᵛ coefficients (for all variables except space). Space random coefficient not estimated to aid estimation. 
X:  2217x6 matrix of observables (including price)
s:  2217x1 vector of product market shares
Z:  2217x15 vector of BLP instruments
v:  50x5 vector of random draws from joint normal mean 0 variance 1. 
    One random draw per each of the 5 θ₂ coefficents per person.
market_id: 2217x1 vector of market id for each product/observation (cdid, market = years in this dataset)

Does not use θ₁ as an input. Rather, backs out θ₁ from θ₂ in the step 2.
This allows for optimization over only the θ₂ coefficients (5) without including θ₁ (6 others).
=#
θ = [0.0, 0.0, 0.0, 0.0, 0.0]

function fjump(θ)
        # run objective function and get key outputs
        θ_matrix = [θ[1], θ[2], θ[3], θ[4], θ[5]]  # Reshape the vector into a one-row matrix
        Q, θ₁, ξ, 𝒯 = demand_objective_function(θ_matrix,X,share,Z,v_50,cdid)
        # return objective function value
        return Q
end

fjump(θ)

function demandgradient(θ)
       # run objective function to update ξ and 𝒯 values for new θ₂
       θ_matrix = [θ[1], θ[2], θ[3], θ[4], θ[5]]  # Reshape the vector into a one-row matrix
       Q, θ₁, ξ, 𝒯 = demand_objective_function(θ_matrix,X,share,Z,v_50,cdid)
       # calculate gradient and record value
       g = gradient(θ_matrix,X,Z,v_50,cdid,ξ,𝒯)
      return g
end 

demandgradient(θ)

JuMP.register(BLPdemand,:fjump,5,fjump,demandgradient;autodiff=false)
#JuMP.register(BLPdemand,:fjump,5,fjump;autodiff=true)

JuMP.@NLobjective(BLPdemand,Min,fjump())

JuMP.optimize!(BLPdemand)
minf=JuMP.objective_value(BLPdemand)