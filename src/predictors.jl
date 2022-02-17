
function predictor_mechanism_out(
    ::Val{false},
    X::AbstractArray{T1,2},
    dbn_data::DBN_Data{T1},
    n::Int,
    active_parents::Vector{<:Real},
    Z::Matrix{<:Real},
    Sigma::Union{Missing,Matrix{<:Real}},
    R::Matrix{<:Real},
)::Matrix{T1} where {T1<:Real}
    X
end
"""
    function predictor_mechanism_out(n, active_parents, Z, X0, X1, Sigma, R)
"""
function predictor_mechanism_out(
    ::Val{true},
    X::AbstractArray{T1,2},
    dbn_data::DBN_Data{T1},
    n::Int,
    active_parents::Vector{<:Real},
    Z::Matrix{<:Real},
    Sigma::Union{Missing,Matrix{<:Real}},
    R::Matrix{<:Real},
)::Matrix{T1} where {T1<:Real}

    X = zeros(T1, n, length(active_parents) * 2)
    counter = 1
    for p in active_parents
        if maximum(Z[:, p]) == 1
            wh1 = Z[:, p] .== 1
            wh2 = Z[:, p] .== 0
            if ismissing(Sigma)
                X[wh1, counter] = crossfun1(dbn_data.X0[wh1, :]) * dbn_data.X1_trans[wh1, p]
                counter += 1
                X[wh2, counter] = crossfun1(dbn_data.X0[wh2, :]) * dbn_data.X1_trans[wh2, p]
                counter += 1
            else
                X[wh1, counter] .=
                    (
                        inv(R[wh1, wh1]) -
                        (R[wh1, wh1] \ dbn_data.X0[wh1, :]) * (
                            crossprod(
                                dbn_data.X0[wh1, :],
                                (Sigma[wh1, wh1] \ dbn_data.X0[wh1, :]),
                            ) \ transpose(Sigma[wh1, wh1] \ dbn_data.X0[wh1, :])
                        )
                    ) * dbn_data.X1_trans[wh1, p]
                counter += 1
                X[wh2, counter] .=
                    (
                        inv(R[wh2, wh2]) -
                        (R[wh2, wh2] \ dbn_data.X0[wh2, :]) * (
                            crossprod(
                                dbn_data.X0[wh2, :],
                                (Sigma[wh2, wh2] \ dbn_data.X0[wh2, :]),
                            ) \ transpose(Sigma[wh2, wh2] \ dbn_data.X0[wh2, :])
                        )
                    ) * dbn_data.X1_trans[wh2, p]
                counter += 1
            end
        else
            X[:, counter] .= dbn_data.X1_trans[:, p]
            counter += 1
        end
    end

    X = counter > 1 ? X[:, 1:counter-1] : X
    X
end


function predictors_mechanismchangein(::Val{false},
    X::Matrix{T},
    dbn_data::DBN_Data{T},
    Z::Matrix{<:Real},
    p_inds::Vector{Int},
    p::Int,
    Sigma::Union{Missing,Matrix{T}},
    R::Matrix{T},
)::Matrix{T} where {T<:Real}
    X
end


function predictors_mechanismchangein(
    ::Val{true},
    X::Matrix{T},
    dbn_data::DBN_Data{T},
    Z::Matrix{<:Real},
    p_inds::Vector{Int},
    p::Int,
    Sigma::Union{Missing,Matrix{T}},
    R::Matrix{T},
)::Matrix{T} where {T<:Real}
    b = length(p_inds)
    Y = zeros(n, 2 * b)
    wh1 = Z[:, p] .== 1
    wh0 = Z[:, p] .== 0
    if ismissing(Sigma)
        Y[wh1, 1:b] = crossfun1(dbn_data.X0[wh1, :]) * dbn_data.X1_trans[wh1, p_inds]
        Y[wh0, b+1:end] = crossfun1(dbn_data.X0[wh0, :]) * dbn_data.X1_trans[wh0, p_inds]
    else
        Y[wh1, 1:b] =
            sigma_mult(R, Sigma, dbn_data.X0[wh1, :], wh1) * dbn_data.X1_trans[wh1, p_inds]
        Y[wh0, (b+1):2*b] =
            sigma_mult(R, Sigma, dbn_data.X0[wh0, :], wh0) * dbn_data.X1_trans[wh0, p_inds]
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
    X::Matrix{R},
    IP0::Matrix{R},
    Z::Matrix{<:Real},
    p_inds::Vector{Int},
)::Matrix{R} where {R<:Real}
    fe = IP0 * Z
    Y = hcat(X, fe[:, union(p_inds, findall(vec(maximum(Z, dims = 1)) .== 1))])
    Y
end

function fixed_effect_out(
    ::Val{false},
    X::Matrix{R},
    IP0::Matrix{R},
    Z::Matrix{<:Real},
    p_inds::Vector{Int},
)::Matrix{R} where {R<:Real}
    X
end

