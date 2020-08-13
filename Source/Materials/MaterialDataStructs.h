#pragma once

#include "RayLib/Vector.h"
#include "TracerLib/GPUMedium.cuh"

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
    const GPUMedium* dMediums;
};

using EmissiveMatData = AlbedoMatData;