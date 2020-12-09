#pragma once

#include "RayLib/Vector.h"
#include "GPUMediumI.h"

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

using EmissiveMatData = AlbedoMatData;