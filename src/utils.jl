"""
count the number of possible graphs
"""
function count_graphs(n_nodes::I, maxindegree::I)::I where {I<:Int}
    sum(i -> binomial(n_nodes, i), 1:maxindegree) + 1
end


# provide unified interface to variant of crossprod in R
function crossprod(X::AbstractArray{R, 1}, Y::AbstractArray{R, 1})::R where {R<:Real}
    dot(X, Y)
end


 function crossprod(X::M)::M where {M<:Array{<:Real, 2}}
     transpose(X) * X
 end


function crossprod(X::AbstractArray{M})::M where M <:Real
    crossprod(X, X)
end



# reocurring function
function crossfun1(X0p::AbstractArray{T})::Matrix{T} where {T<:Real}
    crossfun1(X0p, one(T))
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
    ::Val{true},
    data::DBN_Data{R},
    obs::Vector{Int},
    p::Int
)::Parent_struct{R} where {R<:Real}
    n, P = size(data.y)
    #IP0 = Array{eltype(data.y)}(undef, n, n)
    data.R = transpose(cholesky(Sigma).U)
    solvesigma = (data.Sigma[obs, obs] \ data.X0[obs, :])
    IP0 =
        inv(data.R[obs, obs]) -
        (data.R[obs, obs] \ data.X0[obs, :]) *
        (crossprod(data.X0[obs, :], solvesigma) \ transpose(solvesigma))
    
    data.y_trans[obs, p] = IP0 * data.y[obs, p]
    Parent_struct(IP0 * data.y[obs, p], data.X0, IP0*data.X1[obs, p], IP0)
end


"""
    remove X0 from y without prior on Sigma
"""
function disentangle(
    ::Val{false},
    data::DBN_Data{R},
    obs::Vector{Int},
    p::Int
)::Parent_struct{R} where {R<:Real}
    n = size(data.y, 1)
    if any(obs != collect(1:n))
        embed_y = zeros(n)
        embed_x0 = zeros(size(data.X0))
        embed_x1 = zeros(n)
        embed_y[obs] .= data.y[obs, p]
        embed_x0[obs, :] .= data.X0[obs, :]
        embed_x1[obs] .= data.X1[obs, p]
        IP0 = crossfun1(embed_x0)
        #data.y_trans[obs, p] = IP0 * data.y[obs, p]
        Parent_struct(vec(IP0 * embed_y), embed_x0, vec(IP0*embed_x1), IP0, obs)
    else
        
        IP0 = crossfun1(data.X0[obs, :])
        #data.y_trans[obs, p] = IP0 * data.y[obs, p]
        Parent_struct(vec(IP0 * data.y[obs, p]), data.X0[obs, :], vec(IP0*data.X1[obs, p]), IP0, obs)
    end
end



function Xfun(
    ::Val{false},
    X::Matrix{R},
    X0::Matrix{R},
    dbn_data::DBN_Data{R},
    obs::Vector{Int},
)::Matrix{R} where {R<:Real}

    crossfun1(X0) * X
end


function Xfun(
    ::Val{true},
    X::Matrix{R},
    X0::Matrix{R},
    dbn_data::DBN_Data{R},
    obs::Vector{Int}
)::Matrix{R} where {R<:Real}
    sigma_mult(dbn_data.R, dbn_data.Sigma, X0, obs) * X
end
