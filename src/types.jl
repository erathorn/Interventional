

struct Parent_struct{T<:Real, N}
    y::Vector{T}
    X0::Matrix{T}
    X1::Array{T, N}
    IP0::Matrix{T}
    obs::Vector{Int}

    function Parent_struct(y::Vector{T}, X0::Matrix{T}, X1::Vector{T}) where {T<:Real}
        new{T, 1}(y, X0, X1, Matrix{T}(undef, length(y), length(y)), length(y))
    end

    function Parent_struct(y::Vector{T}, X0::Matrix{T}, X1::Vector{T}, IP0::Matrix{T}) where {T<:Real}
        new{T, 1}(y, X0, X1, IP0, length(y))
    end

    function Parent_struct(y::Vector{T}, X0::Matrix{T}, X1::Array{T,N}, IP0::Matrix{T}) where {T<:Real, N}
        new{T, N}(y, X0, X1, IP0, length(y))
    end

    function Parent_struct(y::Vector{T}, X0::Matrix{T}, X1::Vector{T}, IP0::Matrix{T}, obs::Vector{Int}) where {T<:Real}
        new{T, 1}(y, X0, X1, IP0, obs)
    end

    function Parent_struct(y::Vector{T}, X0::Matrix{T}, X1::Array{T,N}, IP0::Matrix{T}, obs::Vector{Int}) where {T<:Real, N}
        new{T, N}(y, X0, X1, IP0, obs)
    end
end


function Parent_struct(p::Parent_struct{T, M}, X::Array{T, N})::Parent_struct{T, N} where {T<:Real, M, N}
    Parent_struct(p.y, p.X0, X, p.IP0, p.obs)
end

Base.size(p::Parent_struct) = size(p.X1)
struct DBN_Data{T<:Real}
    y::Matrix{T}
    y_trans::Matrix{T}
    X0::Matrix{T}
    X0_trans::Matrix{T}
    X1::Matrix{T}
    X1_trans::Matrix{T}
    Sigma::Matrix{T}
    R::Matrix{T}

    function DBN_Data(y::Matrix{T}, X0::Matrix{T}, X1::Matrix{T}, Sigma::Matrix{T}) where {T<:Real}
        new{T}(y, y, X0, X0, X1, X1, Sigma, Sigma)
    end
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
