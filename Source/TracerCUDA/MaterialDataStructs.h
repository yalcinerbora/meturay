#pragma once

#include "RayLib/Vector.h"
#include "GPUMediumI.h"
#include "SamplerI.cuh"

struct NullData {};

struct AlbedoMatData
{
    const Vector3* dAlbedo;
};

struct ReflectMatData
{
    const Vector4* dAlbedoAndRoughness;
};

struct RefractMatData
{
    const Vector3* dAlbedo;
    const uint32_t* mediumIndices;
    
    // Global Medium Array pointer
    const GPUMediumI* const* dMediums;
    uint32_t baseMediumIndex;
};

struct UnrealMatData
{
    const SamplerCF<2, 3, float>* dAlbedo;
    const SamplerCF<2, 1, float>* dRoughness;
    const SamplerCF<2, 1, float>* dMetallic;
    const SamplerI<2, 3, float>** dNormal;
};

struct LambertTMatData
{
    const SamplerCF<2, 3, float>* dAlbedo;
    const SamplerI<2, 3, float>** dNormal;
};

using EmissiveMatData = AlbedoMatData;