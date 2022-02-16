

"""
    function predictor_mechanism_out(n, active_parents, Z, X0, X1, Sigma, R)
"""
function predictor_mechanism_out(
    n::Int,
    active_parents::Vector{<:Real},
    Z::Matrix{<:Real},
    X0::M,
    X1::M,
    Sigma::Union{Missing,Matrix{<:Real}},
    R::Matrix{<:Real},
)::M where {M<:Matrix{<:Real}}

    X = zeros(n, length(active_parents) * 2)
    counter = 1
    for p in active_parents
        if maximum(Z[:, p]) == 1
            wh1 = Z[:, p] .== 1
            wh2 = Z[:, p] .== 0
            if ismissing(Sigma)
                X[wh1, counter] = crossfun1(X0[wh1, :]) * X1[wh1, p]
                counter += 1
                X[wh2, counter] = crossfun1(X0[wh2, :]) * X1[wh2, p]
                counter += 1
            else
                X[wh1, counter] .=
                    (
                        inv(R[wh1, wh1]) -
                        (R[wh1, wh1] \ X0[wh1, :]) * (
                            crossprod(X0[wh1, :], (Sigma[wh1, wh1] \ X0[wh1, :])) \
                            transpose(Sigma[wh1, wh1] \ X0[wh1, :])
                        )
                    ) * X1[wh1, p]
                counter += 1
                X[wh2, counter] .=
                    (
                        inv(R[wh2, wh2]) -
                        (R[wh2, wh2] \ X0[wh2, :]) * (
                            crossprod(X0[wh2, :], (Sigma[wh2, wh2] \ X0[wh2, :])) \
                            transpose(Sigma[wh2, wh2] \ X0[wh2, :])
                        )
                    ) * X1[wh2, p]
                counter += 1
            end
        else
            X[:, counter] .= X1[:, p]
            counter += 1
        end
    end

    X = counter > 1 ? X[:, 1:counter-1] : X
    X
end



function predictors_mechanismchangein(X1_fun, X0, Z, p_inds, p, Sigma, R)
    b = length(p_inds)
    X = zeros(n, 2 * b)
    wh1 = Z[:, p] .== 1
    wh0 = Z[:, p] .== 0
    if ismissing(Sigma)
        X[wh1, 1:b] = crossfun1(X0[wh1, :]) * X1_fun[wh1, p_inds]
        X[wh0, b+1:end] = crossfun1(X0[wh0, :]) * X1_fun[wh0, p_inds]
    else
        X[wh1, 1:b] = sigma_mult(R, Sigma, X0[wh1, :], wh1) * X1_fun[wh1, p_inds]
        X[wh0, (b+1):2*b] = sigma_mult(R, Sigma, X0[wh0, :], wh0) * X1_fun[wh0, p_inds]
    end
    X
end

function perfectout!(X, X1_fun, X0, Z, p_inds, Sigma, R)
    for pa in p_inds
        if maximum(Z[obs, pa]) == 1
            wh0 = obs[Z[obs, pa].==0]
            X0p = X0[wh0, :]
            if ismissing(Sigma)
                X[wh0, p_inds.==pa] = crossfun1(X0p) * X1_fun[wh0, pa]
            else
                X[wh0, p_inds.==pa] = sigma_mult(R, Sigma, X0p, wh0) * X1[wh0, pa]
            end
            X[Z[obs, pa]==1, p_inds.==pa] .= 0
        end
    end
end

