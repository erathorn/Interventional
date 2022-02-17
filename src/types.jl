

struct DBN_Data{R<:Real}
    y::Matrix{R}
    y_trans::Matrix{R}
    X0::Matrix{R}
    X0_trans::Matrix{R}
    X1::Matrix{R}
    X1_trans::Matrix{R}


    function DBN_Data(y::Matrix{R}, X0::Matrix{R}, X1::Matrix{R}) where {R<:Real}
        new{R}(y, y, X0, X0, X1, X1)
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
