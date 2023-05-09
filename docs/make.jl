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
    authors="Guillaume Dalle <22795598+gdalle@users.noreply.github.com> and contributors",
    repo="https://github.com/gdalle/DifferentiableFrankWolfe.jl/blob/{commit}{path}#{line}",
    sitename="DifferentiableFrankWolfe.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gdalle.github.io/DifferentiableFrankWolfe.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md", "Tutorial" => "tutorial.md"],
    linkcheck=true,
)

deploydocs(; repo="github.com/gdalle/DifferentiableFrankWolfe.jl", devbranch="main")
