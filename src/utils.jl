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
function crossfun1(X0p::AbstractArray{T, 2})::Matrix{T} where {T<:Real}
    crossfun1(X0p, one(T))
end

function crossfun1(X0p::AbstractArray{T, 2}, scale::T)::Matrix{T} where {T<:Real}
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
    data::DBN_Data{R},
    obs,
    Sigma::Matrix{R},
)::Tuple{Matrix{R}, Matrix{R}} where {R<:Real}
    n, P = size(data.y)
    IP0 = Array{eltype(data.y)}(undef, n, n)
    R1 = transpose(cholesky(Sigma).U)
    solvesigma = (Sigma[obs, obs] \ data.X0[obs, :])
    IP0 =
        inv(R[obs, obs]) -
        (R[obs, obs] \ data.X0[obs, :]) * (crossprod(data.X0[obs, :], solvesigma) \ transpose(solvesigma))
    @inbounds for i = 1:P
        data.y_trans[obs, i] = IP0 * data.y[obs, i]
    end
    IP0, R1
end


"""
    remove X0 from y without prior on Sigma
"""
function disentangle(
    data::DBN_Data{R},
    obs,
    Sigma::Missing,
)::Tuple{Matrix{R}, Matrix{R}} where R<:Real
    n, P = size(data.y)
    R1 = Matrix{eltype(data.y)}(undef, 1, 1)
    IP0 = crossfun1(data.X0[obs, :])
    @inbounds for i = 1:P
        data.y_trans[obs, i] = IP0 * data.y[obs, i]
    end
    IP0, R1
end





