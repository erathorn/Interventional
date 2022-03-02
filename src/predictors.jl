
function predictor_mechanism_out(
    ::Val{false},
    dbn_data::DBN_Data{T1},
    parent_data::Parent_struct{T1, N},
    p::Int,
    Z::Matrix{<:Real},
    covariance::Bool,
)::Parent_struct{T1, N} where {T1<:Real, N}
    parent_data
end
"""
    function predictor_mechanism_out(n, active_parents, Z, X0, X1, Sigma, R)
"""
function predictor_mechanism_out(
    ::Val{true},
    dbn_data::DBN_Data{T1},
    parent_data::Parent_struct{T1, N},
    p::Int,
    Z::Matrix{<:Real},
    covariance::Bool,
)::Parent_struct{T1, 2} where {T1<:Real, N}

    X = zeros(T1, size(Z, 1), 2)
    
    if maximum(Z[:, p]) == 1
        wh1 = Z[:, p] .== 1
        wh2 = Z[:, p] .== 0
        if !covariance
            X[wh1, 1] = crossfun1(parent_data.X0[wh1,:]) * parent_data.X1[wh1]
            
            X[wh2, 2] = crossfun1(parent_data.X0[wh2,:]) * parent_data.X1[wh2]
            
        else
            X[wh1, 1] .=
                (
                    inv(dbn_data.R[wh1, wh1]) -
                    (dbn_data.R[wh1, wh1] \ parent_data.X0[wh1, :]) * (
                        crossprod(
                            parent_data.X0[wh1, :],
                            (dbn_data.Sigma[wh1, wh1] \ parent_data.X0[wh1, :]),
                        ) \ transpose(dbn_data.Sigma[wh1, wh1] \ parent_data.X0[wh1, :])
                    )
                ) * parent_data.X1[wh1, p]
            
            X[wh2, 2] .=
                (
                    inv(dbn_data.R[wh2, wh2]) -
                    (dbn_data.R[wh2, wh2] \ parent_data.X0[wh2, :]) * (
                        crossprod(
                            parent_data.X0[wh2, :],
                            (dbn_data.Sigma[wh2, wh2] \ parent_data.X0[wh2, :]),
                        ) \ transpose(dbn_data.Sigma[wh2, wh2] \ parent_data.X0[wh2, :])
                    )
                ) * parent_data.X1[wh2, p]
            
        end
    else
        X = parent_data.X1
    end

    Parent_struct(parent_data, X)
end


function predictors_mechanismchangein(::Val{false},
    parent::Parent_struct{T},
    X::Matrix{T},
    dbn_data::DBN_Data{T},
    Z::Matrix{<:Real},
    p::Int
)::Matrix{T} where {T<:Real}
    X
end


function predictors_mechanismchangein(::Val{true},
    parent::Parent_struct{T},
    X::Matrix{T},
    dbn_data::DBN_Data{T},
    Z::Matrix{<:Real},
    p::Int
)::Matrix{T} where {T<:Real}
    b = size(X, 2)
    Y = zeros(n, 2 * b)
    wh1 = Z[:, p] .== 1
    wh0 = Z[:, p] .== 0
    if ismissing(Sigma)
        Y[wh1, 1:b] = crossfun1(X0[wh1, :]) * X1[wh1, :]
        Y[wh0, b+1:end] = crossfun1(X0[wh0, :]) * X1[wh0, :]
    else
        Y[wh1, 1:b] =
            sigma_mult(R, Sigma, parent.X0[wh1, :], wh1) * parent.X1[wh1,:]
        Y[wh0, (b+1):2*b] =
            sigma_mult(R, Sigma, parent.X0[wh0, :], wh0) * parent.X1[wh0, :]
    end
    Y
end

function perfectout(
    ::Val{false},
    X::Matrix{T},
    dbn_data::DBN_Data{T},
    Z::Matrix{<:Real},
    p_inds::Vector{Int},
    Sigma::Union{Missing,Matrix{T}},
    R::Matrix{T},
)::Matrix{T} where {T<:Real}
    X
end

function perfectout(
    ::Val{true},
    X::Matrix{T},
    dbn_data::DBN_Data{T},
    Z::Matrix{<:Real},
    p_inds::Vector{Int},
    Sigma::Union{Missing,Matrix{T}},
    R::Matrix{T},
)::Matrix{T} where {T<:Real}
    Y = deepcopy(X)
    for pa in p_inds
        if maximum(Z[obs, pa]) == 1
            wh0 = obs[Z[obs, pa].==0]
            if ismissing(Sigma)
                Y[wh0, p_inds.==pa] =
                    crossfun1(dbn_data.X0[wh0, :]) * dbn_data.X1_trans[wh0, pa]
            else
                Y[wh0, p_inds.==pa] =
                    sigma_mult(R, Sigma, dbn_data.X0[wh0, :], wh0) *
                    dbn_data.X1_trans[wh0, pa]
            end
            Y[Z[obs, pa]==1, p_inds.==pa] .= 0
        end
    end
    Y
end

function fixed_effect_out(
    ::Val{true},
    parent_data::Parent_struct{R, N},
    Z::Matrix{<:Real},
    p::Int,
)::Parent_struct{R} where {R<:Real, N}
    fe = parent_data.IP0 * Z
    Y = hcat(parent_data.X1, fe[:, union(p, findall(vec(maximum(Z, dims = 1)) .== 1))])
    Parent_struct(parent_data, Y)
end

function fixed_effect_out(
    ::Val{false},
    parent_data::Parent_struct{R,N},
    Z::Matrix{<:Real},
    p::Int,
)::Parent_struct{R,N} where {R<:Real, N}
    parent_data
end

