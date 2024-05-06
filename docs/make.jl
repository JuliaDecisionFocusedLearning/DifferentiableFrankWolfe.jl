using DifferentiableFrankWolfe
using Documenter
using Literate

DocMeta.setdocmeta!(
    DifferentiableFrankWolfe,
    :DocTestSetup,
    :(using DifferentiableFrankWolfe);
    recursive=true,
)

Literate.markdown(
    joinpath(@__DIR__, "..", "examples", "tutorial.jl"),
    joinpath(@__DIR__, "src");
    documenter=true,
    flavor=Literate.DocumenterFlavor(),
)

makedocs(;
    modules=[DifferentiableFrankWolfe],
    authors="Guillaume Dalle",
    sitename="DifferentiableFrankWolfe.jl",
    format=Documenter.HTML(;
        canonical="https://JuliaDecisionFocusedLearning.github.io/DifferentiableFrankWolfe.jl",
        edit_link="main",
    ),
    pages=["Home" => "index.md", "Tutorial" => "tutorial.md"],
)

deploydocs(;
    repo="github.com/JuliaDecisionFocusedLearning/DifferentiableFrankWolfe.jl",
    devbranch="main",
)
