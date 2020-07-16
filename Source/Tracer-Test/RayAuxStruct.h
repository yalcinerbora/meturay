#pragma once

#include <cstdint>
#include "RayLib/Vector.h"

#include <cuda_fp16.h>

enum class RayType : uint8_t
{
    NEE_RAY,
    TRANS_RAY,
    PATH_RAY,
    CAMERA_RAY
};

struct RayAuxBasic
{
    // Pixel index
    uint32_t        pixelIndex;
};

struct RayAuxPath
{
    // Path throughput 
    // (a.k.a. total radiance coefficient along the path)
    Vector3f        radianceFactor;
    
    uint32_t        pixelIndex;     // Starting pixel index of the ray
    uint32_t        endPointIndex;  // Destination of the ray if applicable (i.e. NEE Ray)
    uint16_t        mediumIndex;    // Current Medium of the Ray
    uint8_t         depth;          // Current path depth    
    RayType         type;           // Ray Type
};

static constexpr int asd = sizeof(RayAuxPath);

//static_cast(sizeof(RayAuxPath) == 10 * sizeof(float), "");

static const RayAuxBasic InitialBasicAux = RayAuxBasic
{ 
    0
};

static const RayAuxPath InitialPathAux = RayAuxPath
{
    Vector3f(1.0f, 1.0f, 1.0f),
    0, 0, 0,
    1,
    RayType::CAMERA_RAY
};