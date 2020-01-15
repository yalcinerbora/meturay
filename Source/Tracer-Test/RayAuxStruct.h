#pragma once

#include <cstdint>
#include "RayLib/Vector.h"

struct RayAuxBasic
{
    Vector3f        radianceFactor;
    uint32_t        pixelId;
    uint32_t        pixelSampleId;
    uint8_t         depth;
    bool            lightRay;
};


struct RayAuxVolume
{
    Vector3f        radianceFactor;
    uint32_t        pixelId;
    uint32_t        pixelSampleId;
    uint8_t         depth;
    bool            lightRay;
};