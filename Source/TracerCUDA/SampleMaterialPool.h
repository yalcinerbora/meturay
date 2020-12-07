#pragma once

#include "TracerCUDALib/TracerLogicPools.h"

class SampleMaterialPool final : public MaterialLogicPoolI
{
    public:
        // Constructors & Destructor
        SampleMaterialPool();
        ~SampleMaterialPool() = default;
};

extern "C" _declspec(dllexport) MaterialLogicPoolI * __stdcall GenerateSampleMaterialPool();

extern "C" _declspec(dllexport) void __stdcall DeleteSampleMaterialPool(MaterialLogicPoolI*);