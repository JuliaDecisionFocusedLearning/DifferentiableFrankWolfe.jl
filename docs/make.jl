using DifferentiableFrankWolfe
using Documenter

DocMeta.setdocmeta!(DifferentiableFrankWolfe, :DocTestSetup, :(using DifferentiableFrankWolfe); recursive=true)

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
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/gdalle/DifferentiableFrankWolfe.jl",
    devbranch="main",
)
