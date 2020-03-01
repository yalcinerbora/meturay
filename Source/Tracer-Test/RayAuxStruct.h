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
    // Path throughput
    Vector3f        radianceFactor;
    // Pixel index
    uint32_t        pixelId; 
    // Current medium index
    half            mediumIndex;
    // Current path depth
    uint8_t         depth;
    // Ray Type
    RayType         type;
};

static const RayAuxBasic InitialBasicAux = RayAuxBasic
{ 
    Vector3f(1.0f, 1.0f, 1.0f),
    0, 
    half{}, 
    1, 
    RayType::CAMERA_RAY
};