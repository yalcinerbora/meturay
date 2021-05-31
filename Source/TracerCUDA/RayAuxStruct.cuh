#pragma once

#include <cstdint>
#include "RayLib/Vector.h"

#include <cuda_fp16.h>

struct RayReg;

enum class RayType : uint8_t
{
    NEE_RAY,
    SPECULAR_PATH_RAY,
    PATH_RAY,
    CAMERA_RAY
};

struct RayAuxBasic
{
    uint32_t        pixelIndex;
};

struct RayAuxAO
{
    Vector3f        aoFactor;
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

struct RayAuxPPG
{
    // Path throughput
    // (a.k.a. total radiance coefficient along the path)
    Vector3f        radianceFactor;

    uint32_t        pixelIndex;     // Starting pixel index of the ray
    uint32_t        endPointIndex;  // Destination of the ray if applicable (i.e. NEE Ray)
    uint16_t        mediumIndex;    // Current Medium of the Ray
    uint8_t         depth;          // Current path depth
    RayType         type;           // Ray Type
    uint8_t         pathIndex;      // Local path node index
};

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

static const RayAuxAO InitialAOAux = RayAuxAO
{
    Vector3f(1.0f, 1.0f, 1.0f),
    0
};

__device__ __host__
inline void RayInitBasic(RayAuxBasic& gOutBasic,
                         // Input
                         const RayAuxBasic& defaults,
                         const RayReg& ray,
                         // Index
                         uint16_t mediumIndex,
                         const uint32_t localPixelId,
                         const uint32_t pixelSampleId)
{
    RayAuxBasic init = defaults;
    init.pixelIndex = localPixelId;
    gOutBasic = init;
}

__device__ __host__
inline void RayInitPath(RayAuxPath& gOutPath,
                         // Input
                        const RayAuxPath& defaults,
                        const RayReg& ray,
                        // Index
                        uint16_t medumIndex,
                        const uint32_t localPixelId,
                        const uint32_t pixelSampleId)
{
    RayAuxPath init = defaults;
    init.pixelIndex = localPixelId;
    init.type = RayType::CAMERA_RAY;
    init.mediumIndex = medumIndex;
    init.depth = 1;
    gOutPath = init;
}

__device__ __host__
inline void RayInitAO(RayAuxAO& gOutAO,
                      // Input
                      const RayAuxAO& defaults,
                      const RayReg& ray,
                      // Index
                      uint16_t medumIndex,
                      const uint32_t localPixelId,
                      const uint32_t pixelSampleId)
{
    RayAuxAO init = defaults;
    init.pixelIndex = localPixelId;
    gOutAO = init;
}