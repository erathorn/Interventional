using Documenter
using Interventional

makedocs(
    modules = Interventional,
    sitename="Interventional",
    format = Documenter.HTML(),
    pages = [
        "Index" => "index.md",
        "Docs" => "docs.md"],
   )

deploydocs(
    repo = "github.com/erathorn/Interventional.git",
    devbranch="main"
     )