var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = DifferentiableFrankWolfe","category":"page"},{"location":"#DifferentiableFrankWolfe","page":"Home","title":"DifferentiableFrankWolfe","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for DifferentiableFrankWolfe.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [DifferentiableFrankWolfe]","category":"page"},{"location":"#DifferentiableFrankWolfe.ConditionsFW","page":"Home","title":"DifferentiableFrankWolfe.ConditionsFW","text":"ConditionsFW{F,G,M}\n\nDifferentiable optimality conditions for DiffFW, which rely on a custom simplex_projection implementation.\n\n\n\n\n\n","category":"type"},{"location":"#DifferentiableFrankWolfe.DiffFW","page":"Home","title":"DifferentiableFrankWolfe.DiffFW","text":"DiffFW{F,G,M,A,I}\n\nCallable parametrized wrapper for the Frank-Wolfe algorithm θ -> argmin_{x ∈ C} f(x, θ), which can be differentiated implicitly wrt θ.\n\nReference: https://arxiv.org/abs/2105.15183 (especially section 2 and the end of appendix A).\n\nFields\n\nf::F: function f(x, θ) to minimize wrt x\nf_grad1::G: gradient ∇ₓf(x, θ) of f wrt x\nlmo::M: linear minimization oracle θ -> argmin_{x ∈ C} θᵀx, implicitly defines the convex set C\nalg::A: Frank-Wolfe variante used, defaults to away_frank_wolfe\nimplicit::I: implicit function constructed from the previous fields\n\n\n\n\n\n","category":"type"},{"location":"#DifferentiableFrankWolfe.DiffFW-2","page":"Home","title":"DifferentiableFrankWolfe.DiffFW","text":"DiffFW(f, f_grad1, lmo[, alg=away_frank_wolfe])\n\nConstructor which chooses a default algorithm and creates the implicit function object.\n\n\n\n\n\n","category":"type"},{"location":"#DifferentiableFrankWolfe.DiffFW-Tuple{AbstractArray{<:Real}}","page":"Home","title":"DifferentiableFrankWolfe.DiffFW","text":"dfw(θ; frank_wolfe_kwargs)\n\nApply the Frank-Wolfe algorithm to θ with settings defined by frank_wolfe_kwargs.\n\n\n\n\n\n","category":"method"},{"location":"#DifferentiableFrankWolfe.ForwardFW","page":"Home","title":"DifferentiableFrankWolfe.ForwardFW","text":"ForwardFW{F,G,M,A}\n\nUnderlying solver for DiffFW, which relies on a variant of Frank-Wolfe.\n\n\n\n\n\n","category":"type"},{"location":"#DifferentiableFrankWolfe.simplex_projection-Tuple{AbstractVector{<:Real}}","page":"Home","title":"DifferentiableFrankWolfe.simplex_projection","text":"simplex_projection(z)\n\nCompute the Euclidean projection of the vector z onto the probability simplex.\n\nReference: https://arxiv.org/abs/1602.02068.\n\n\n\n\n\n","category":"method"},{"location":"#DifferentiableFrankWolfe.simplex_projection_and_support-Tuple{AbstractVector{<:Real}}","page":"Home","title":"DifferentiableFrankWolfe.simplex_projection_and_support","text":"simplex_projection_and_support(z)\n\nCompute the Euclidean projection p of z on the probability simplex and the indicators s of its support.\n\nReference: https://arxiv.org/abs/1602.02068.\n\n\n\n\n\n","category":"method"}]
}
