var documenterSearchIndex = {"docs":
[{"location":"#Documentation-for-Interventional","page":"Index","title":"Documentation for Interventional","text":"","category":"section"},{"location":"","page":"Index","title":"Index","text":"This is the documention of the Interventional package. It reimplements the \"InterventionalDBN\" package written in the R programming language. This implementation aims for modularity and speed.","category":"page"},{"location":"","page":"Index","title":"Index","text":"Original R Code: https://cran.r-project.org/web/packages/interventionalDBN/index.html","category":"page"},{"location":"docs/#Documentation","page":"Docs","title":"Documentation","text":"","category":"section"},{"location":"docs/","page":"Docs","title":"Docs","text":"Assemble the docstrings.","category":"page"},{"location":"docs/","page":"Docs","title":"Docs","text":"Modules = [Interventional]\nPages   = [\"src/predictors.jl\", \"src/utils.jl\", \"src/datareader.jl\", \"src/types.jl\"]\nFilter =","category":"page"},{"location":"docs/#Interventional.predictor_mechanism_out-Union{Tuple{N}, Tuple{T}, Tuple{Val{false}, Interventional.DBN_Data{T}, Interventional.Parent_struct{T, N}, Int64, Matrix{<:Real}, Bool}} where {T<:Real, N}","page":"Docs","title":"Interventional.predictor_mechanism_out","text":"function predictormechanismout(     ::Val{false},     dbndata::DBNData{T},     parentdata::Parentstruct{T, N},     p::Int,     Z::Matrix{<:Real},     covariance::Bool, )::Parent_struct{T, N} where {T<:Real, N}\n\npass everything through\n\n\n\n\n\n","category":"method"},{"location":"docs/#Interventional.predictor_mechanism_out-Union{Tuple{N}, Tuple{T}, Tuple{Val{true}, Interventional.DBN_Data{T}, Interventional.Parent_struct{T}, Int64, Matrix{<:Real}, Bool}} where {T<:Real, N}","page":"Docs","title":"Interventional.predictor_mechanism_out","text":"function predictormechanismout(     ::Val{true},     dbndata::DBNData{T},     parentdata::Parentstruct{T, N},     p::Int,     Z::Matrix{<:Real},     covariance::Bool, )::Parent_struct{T, 2} where {T<:Real, N}\n\nassemble predictors for mechanism out change\n\n\n\n\n\n","category":"method"},{"location":"docs/#Interventional.count_graphs-Union{Tuple{I}, Tuple{I, I}} where I<:Int64","page":"Docs","title":"Interventional.count_graphs","text":"count the number of possible graphs\n\n\n\n\n\n","category":"method"},{"location":"docs/#Interventional.crossprod-Tuple{M} where M<:(Matrix{<:Real})","page":"Docs","title":"Interventional.crossprod","text":"function crossprod(X::M)::M where {M<:Array{<:Real,2}}\n\nEmulate the crossprod function in R\n\n\n\n\n\n","category":"method"},{"location":"docs/#Interventional.crossprod-Union{Tuple{AbstractVector{M}}, Tuple{M}} where M<:Real","page":"Docs","title":"Interventional.crossprod","text":"function crossprod(X::AbstractArray{M, 1})::M where {M<:Real}\n\nEmulate the crossprod function in R\n\n\n\n\n\n","category":"method"},{"location":"docs/#Interventional.crossprod-Union{Tuple{R}, Tuple{AbstractVector{R}, AbstractVector{R}}} where R<:Real","page":"Docs","title":"Interventional.crossprod","text":"function crossprod(X::AbstractArray{R,1}, Y::AbstractArray{R,1})::R where {R<:Real}\n\nEmulate the crossprod function in R\n\n\n\n\n\n","category":"method"},{"location":"docs/#Interventional.disentangle-Union{Tuple{R}, Tuple{Val{false}, Interventional.DBN_Data{R}, Vector{Int64}, Int64}} where R<:Real","page":"Docs","title":"Interventional.disentangle","text":"function disentangle(     ::Val{false},     data::DBNData{R},     obs::Vector{Int},     p::Int )::Parentstruct{R} where {R<:Real}\n\nremove X0 from y without prior on Sigma\n\n\n\n\n\n","category":"method"},{"location":"docs/#Interventional.disentangle-Union{Tuple{R}, Tuple{Val{true}, Interventional.DBN_Data{R}, Vector{Int64}, Int64}} where R<:Real","page":"Docs","title":"Interventional.disentangle","text":"function disentangle(     ::Val{true},     data::DBNData{R},     obs::Vector{Int},     p::Int )::Parentstruct{R} where {R<:Real}\n\nremove X0 from y with Sigma\n\n\n\n\n\n","category":"method"},{"location":"docs/#Interventional.format_DBNstyle-Union{Tuple{Matrix{R}}, Tuple{R}} where R<:Real","page":"Docs","title":"Interventional.format_DBNstyle","text":"function formatDBNstyle(     d::Array{R,2};     gradients::Bool = true,     intercept::Bool = true,     initialIntercept::Bool = true, )::DBNData{R} where {R<:Real}\n\ntransform a data array into a DBN_Data struct which can be used for the interventional inference scheme\n\n\n\n\n\n","category":"method"},{"location":"docs/#Interventional.DBN_Data","page":"Docs","title":"Interventional.DBN_Data","text":"struct DBN_Data{T<:Real}     y::Matrix{T}     X0::Matrix{T}     X1::Matrix{T}     Sigma::Matrix{T}     R::Matrix{T} end\n\nThe structure holding the entire data.\n\n\n\n\n\n","category":"type"},{"location":"docs/#Interventional.InterventionPattern","page":"Docs","title":"Interventional.InterventionPattern","text":"InterventionPattern{B}\n\nHolds the information about the Interventional setup.\n\n\n\n\n\n","category":"type"},{"location":"docs/#Interventional.Parent_struct","page":"Docs","title":"Interventional.Parent_struct","text":"struct Parent_struct{T<:Real, N}     y::Vector{T}     X0::Matrix{T}     X1::Array{T, N}     IP0::Matrix{T}     obs::Vector{Int} end\n\nStructure holding the transforemd data for each node in the DBN.\n\n\n\n\n\n","category":"type"}]
}