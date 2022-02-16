"""
count the number of possible graphs
"""
function count_graphs(n_nodes::I, maxindegree::I)::I where {I<:Int}
    sum(i -> binomial(n_nodes, i), 1:maxindegree) + 1
end


# provide unified interface to variant of crossprod in R
function crossprod(X::AbstractArray{R}, Y::AbstractArray{R})::R where {R<:Real}
    dot(X, Y)
end


function crossprod(X::M)::M where {M<:Array{<:Real}}
    transpose(X) * X
end


function crossprod(X::AbstractArray{M})::M where {M<:Real}
    crossprod(X, X)
end



# reocurring function
function crossfun1(X0p::M)::M where {M<:Array{<:Real}}
    crossfun1(X0p, one(eltype(X0p)))
end

function crossfun1(X0p::AbstractArray{T}, scale::T)::Matrix{T} where {T<:Real}
    I - scale .* X0p * inv(crossprod(X0p)) * transpose(X0p)
end


function sigma_mult(
    R::Matrix{T},
    Sigma::Matrix{T},
    X0p::Matrix{T},
    wh1::Vector{Int},
)::Matrix{T} where {T<:Real}
    inv(R[wh1, wh1]) -
    (R[wh1, wh1] \ X0p) *
    (crossprod(X0p, (Sigma[wh1, wh1] \ X0p)) \ transpose(Sigma[wh1, wh1] \ X0p))

end


# data manipulation


"""
    remove X0 from y with Sigma
"""
function disentangle(
    y::A,
    X0::A,
    obs,
    Sigma::Array{<:Real},
)::Tuple{A,A,A} where {A<:Array{<:Real}}
    n, P = size(y)
    Y_fun = zeros(eltype(y), size(y))
    IP0 = Array{eltype(y)}(undef, n, n)
    R = transpose(cholesky(Sigma).U)
    X0p = X0[obs, :]
    solvesigma = (Sigma[obs, obs] \ X0p)
    IP0 =
        inv(R[obs, obs]) -
        (R[obs, obs] \ X0p) * (crossprod(X0p, solvesigma) \ transpose(solvesigma))
    @inbounds for i = 1:P
        Y_fun[obs, i] = IP0 * y[obs, i]
    end
    Y_fun, IP0, R
end


"""
    remove X0 from y without prior on Sigma
"""
function disentangle(
    y::A,
    X0::A,
    obs,
    Sigma::Missing,
)::Tuple{A,A,A} where {A<:Array{<:Real}}
    n, P = size(y)
    Y_fun = zeros(eltype(y), size(y))
    X0p = X0[obs, :]
    R = Matrix{eltype(y)}(undef, 1, 1)
    IP0 = crossfun1(X0p)
    @inbounds for i = 1:P
        Y_fun[obs, i] = IP0 * y[obs, i]
    end
    Y_fun, IP0, R
end



struct InterventionPattern{B}
    allowSelfEdges::B
    perfectOut::B
    perfectIn::B
    fixedEffectOut::B
    fixedEffectIn::B
    mechanismChangeOut::B
    mechanismChangeIn::B

    function InterventionPattern(;
        perfectOut::Bool = false,
        allowSelfEdges::Bool = false,
        perfectIn::Bool = false,
        fixedEffectIn::Bool = false,
        fixedEffectOut::Bool = false,
        mechanismChangeIn::Bool = false,
        mechanismChangeOut::Bool = false,
    )
        @assert !((perfectOut || perfectIn) && (mechanismChangeIn || mechanismChangeOut)) "mechanism change and perfect interventions cannot be used togehter"
    
        new{Bool}(
            allowSelfEdges,
            perfectOut,
            perfectIn,
            fixedEffectOut,
            fixedEffectIn,
            mechanismChangeOut,
            mechanismChangeIn,
        )
    end
    
end



