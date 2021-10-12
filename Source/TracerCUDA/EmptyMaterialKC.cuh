#pragma once

#include "MaterialDataStructs.h"
#include "MetaMaterialFunctions.cuh"
#include "GPUSurface.h"

class RandomGPU;

template <class Surface>
struct EmptyMatFuncs
{
    static constexpr auto& Sample       = SampleEmpty<NullData, Surface>;
    static constexpr auto& Evaluate     = EvaluateEmpty<NullData, Surface>;
    static constexpr auto& Pdf          = PdfZero<NullData, Surface>;
    static constexpr auto& Emit         = EmitEmpty<NullData, Surface>;
    static constexpr auto& IsEmissive   = IsEmissiveFalse<NullData>;
    static constexpr auto& Specularity  = SpecularityPerfect<NullData, Surface>;
};