var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = DifferentiableFrankWolfe","category":"page"},{"location":"#DifferentiableFrankWolfe","page":"Home","title":"DifferentiableFrankWolfe","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for DifferentiableFrankWolfe.jl.","category":"page"},{"location":"#Public-API","page":"Home","title":"Public API","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [DifferentiableFrankWolfe]\nPrivate = false","category":"page"},{"location":"#DifferentiableFrankWolfe.DifferentiableFrankWolfe","page":"Home","title":"DifferentiableFrankWolfe.DifferentiableFrankWolfe","text":"DifferentiableFrankWolfe\n\nDifferentiable wrapper for FrankWolfe.jl convex optimization routines.\n\n\n\n\n\n","category":"module"},{"location":"#DifferentiableFrankWolfe.DiffFW","page":"Home","title":"DifferentiableFrankWolfe.DiffFW","text":"DiffFW\n\nCallable parametrized wrapper for the Frank-Wolfe algorithm to solve θ -> argmin_{x ∈ C} f(x, θ), which can be differentiated implicitly wrt θ.\n\nReference: https://arxiv.org/abs/2105.15183 (section 2 + end of appendix A).\n\nFields\n\nf: function f(x, θ) to minimize wrt x\nf_grad1: gradient ∇ₓf(x, θ) of f wrt x\nlmo: linear minimization oracle θ -> argmin_{x ∈ C} θᵀx from [FrankWolfe.jl], implicitly defines the convex set C\nalg: optimization algorithm from FrankWolfe.jl\nimplicit: implicit function from ImplicitDifferentiation.jl\n\n\n\n\n\n","category":"type"},{"location":"#DifferentiableFrankWolfe.DiffFW-2","page":"Home","title":"DifferentiableFrankWolfe.DiffFW","text":"(dfw::DiffFW)(θ::AbstractArray, frank_wolfe_kwargs::NamedTuple)\n\nApply the Frank-Wolfe algorithm to θ with settings defined by the named tuple frank_wolfe_kwargs (given as a positional argument).\n\nReturn a couple (x, stats) where x is the solution and stats is a named tuple containing additional information (its contents are not covered by public API, and mostly useful for debugging).\n\n\n\n\n\n","category":"type"},{"location":"#DifferentiableFrankWolfe.DiffFW-Union{Tuple{A}, Tuple{L}, Tuple{G}, Tuple{F}, Tuple{F, G, L}, Tuple{F, G, L, A}} where {F, G, L, A}","page":"Home","title":"DifferentiableFrankWolfe.DiffFW","text":"DiffFW(f, f_grad1, lmo, alg=away_frank_wolfe; implicit_kwargs=(;))\n\nConstructor for DiffFW which chooses a default algorithm and creates the implicit function automatically.\n\n\n\n\n\n","category":"method"},{"location":"#Private-API","page":"Home","title":"Private API","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [DifferentiableFrankWolfe]\nPublic = false","category":"page"},{"location":"#DifferentiableFrankWolfe.ConditionsFW","page":"Home","title":"DifferentiableFrankWolfe.ConditionsFW","text":"ConditionsFW\n\nDifferentiable optimality conditions for DiffFW, which rely on a custom simplex_projection implementation.\n\n\n\n\n\n","category":"type"},{"location":"#DifferentiableFrankWolfe.ForwardFW","page":"Home","title":"DifferentiableFrankWolfe.ForwardFW","text":"ForwardFW\n\nUnderlying solver for DiffFW, which relies on a variant of Frank-Wolfe.\n\n\n\n\n\n","category":"type"},{"location":"#DifferentiableFrankWolfe.simplex_projection-Tuple{AbstractVector{<:Real}}","page":"Home","title":"DifferentiableFrankWolfe.simplex_projection","text":"simplex_projection(z)\n\nCompute the Euclidean projection of the vector z onto the probability simplex.\n\nThis function is differentiable thanks to a custom chain rule.\n\nReference: https://arxiv.org/abs/1602.02068.\n\n\n\n\n\n","category":"method"},{"location":"#DifferentiableFrankWolfe.simplex_projection_and_support-Tuple{AbstractVector{<:Real}}","page":"Home","title":"DifferentiableFrankWolfe.simplex_projection_and_support","text":"simplex_projection_and_support(z)\n\nCompute the Euclidean projection p of z on the probability simplex as well as the indicators s of its support, which are useful for differentiation.\n\nReference: https://arxiv.org/abs/1602.02068.\n\n\n\n\n\n","category":"method"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"EditURL = \"../../examples/tutorial.jl\"","category":"page"},{"location":"tutorial/#Tutorial","page":"Tutorial","title":"Tutorial","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Necessary imports","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using DifferentiableFrankWolfe: DiffFW, simplex_projection\nusing ForwardDiff: ForwardDiff\nusing FrankWolfe: UnitSimplexOracle\nusing Test: @test\nusing Zygote: Zygote","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Constructing the wrapper","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"f(x, θ) = 0.5 * sum(abs2, x - θ)  # minimizing the squared distance...\nf_grad1(x, θ) = x - θ\nlmo = UnitSimplexOracle(1.0)  # ... to the probability simplex\ndfw = DiffFW(f, f_grad1, lmo);  # ... is equivalent to a simplex projection\nnothing #hide","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Calling the wrapper","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"θ = rand(10)","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"frank_wolfe_kwargs = (; max_iteration=100, epsilon=1e-4)\ny, stats = dfw(θ, frank_wolfe_kwargs)\ny","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"y_true = simplex_projection(θ)\n@test Vector(y) ≈ Vector(y_true) atol = 1e-3","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Differentiating the wrapper","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"J1 = Zygote.jacobian(_θ -> dfw(_θ, frank_wolfe_kwargs)[1], θ)[1]\nJ1_true = Zygote.jacobian(simplex_projection, θ)[1]\n@test J1 ≈ J1_true atol = 1e-3","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"J2 = ForwardDiff.jacobian(_θ -> dfw(_θ, frank_wolfe_kwargs)[1], θ)\nJ2_true = ForwardDiff.jacobian(simplex_projection, θ)\n@test J2 ≈ J2_true atol = 1e-3","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"This page was generated using Literate.jl.","category":"page"}]
}
