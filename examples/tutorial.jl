# # Tutorial

# Necessary imports

using DifferentiableFrankWolfe: DiffFW, simplex_projection
using FrankWolfe: UnitSimplexOracle
using Test: @test
using Zygote: jacobian

# Constructing the wrapper

f(x, θ) = 0.5 * sum(abs2, x - θ)  # minimizing the squared distance...
f_grad1(x, θ) = x - θ
lmo = UnitSimplexOracle(1.0)  # ... to the probability simplex
dfw = DiffFW(f, f_grad1, lmo);  # ... is equivalent to a simplex projection

# Calling the wrapper

θ = rand(10)
frank_wolfe_kwargs = (max_iteration=100, epsilon=1e-4)

y = dfw(θ; frank_wolfe_kwargs)
y_true = simplex_projection(θ)
@test Vector(y) ≈ Vector(y_true) atol = 1e-3

# Differentiating the wrapper

J = jacobian(_θ -> dfw(_θ; frank_wolfe_kwargs), θ)[1]
J_true = jacobian(simplex_projection, θ)[1]
@test J ≈ J_true atol = 1e-3
