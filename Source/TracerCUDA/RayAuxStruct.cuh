#pragma once

#include <cstdint>
#include "RayLib/Vector.h"

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
    
    uint32_t        pathIndex;      // Global path node index
};

static const RayAuxBasic InitialBasicAux = RayAuxBasic
{
    0
};

static const RayAuxPath InitialPathAux = RayAuxPath
{
    Vector3f(1.0f, 1.0f, 1.0f),
    0, 0, 0,
    0,
    RayType::CAMERA_RAY
};

static const RayAuxAO InitialAOAux = RayAuxAO
{
    Vector3f(1.0f, 1.0f, 1.0f),
    0
};

static const RayAuxPPG InitialPPGAux = RayAuxPPG
{
    Vector3f(1.0f, 1.0f, 1.0f),
    0, 0, 0,
    0,
    RayType::CAMERA_RAY,
    0
};

class RayAuxInitBasic
{
     private:
         RayAuxBasic defaultValue;

    public:
        RayAuxInitBasic(const RayAuxBasic& aux)
            : defaultValue(aux)
        {}

        __device__ __host__ HYBRID_INLINE
        void operator()(RayAuxBasic& gOutBasic,
                        // Input    
                        const RayReg& ray,
                        // Index
                        uint16_t medumIndex,
                        const uint32_t localPixelId,
                        const uint32_t pixelSampleId) const
        {
            RayAuxBasic init = defaultValue;
            init.pixelIndex = localPixelId;
            gOutBasic = init;
        }
};

class RayAuxInitPath
{
     private:
        RayAuxPath defaultValue;

    public:
        RayAuxInitPath(const RayAuxPath& aux)
            : defaultValue(aux)
        {}

        __device__ __host__ HYBRID_INLINE
        void operator()(RayAuxPath& gOutPath,
                        // Input    
                        const RayReg& ray,
                        // Index
                        uint16_t medumIndex,
                        const uint32_t localPixelId,
                        const uint32_t pixelSampleId) const
        {
            RayAuxPath init = defaultValue;
            init.pixelIndex = localPixelId;
            init.type = RayType::CAMERA_RAY;
            init.mediumIndex = medumIndex;
            init.depth = 1;
            gOutPath = init;
        }
};

class RayAuxInitAO
{
    private:
        RayAuxAO defaultValue;

    public:
        RayAuxInitAO(const RayAuxAO& aux)
            : defaultValue(aux)
        {}

        __device__ __host__ HYBRID_INLINE
        void operator()(RayAuxAO& gOutAO,
                        // Input    
                        const RayReg& ray,
                        // Index
                        uint16_t medumIndex,
                        const uint32_t localPixelId,
                        const uint32_t pixelSampleId) const
        {
            RayAuxAO init = defaultValue;
            init.pixelIndex = localPixelId;
            gOutAO = init;
        }
};

class RayAuxInitPPG
{
    private:
        RayAuxPPG   defaultValue;
        uint32_t    samplePerPixel;

    public:
        RayAuxInitPPG(const RayAuxPPG& aux, 
                      uint32_t samplePerPixel)
            : defaultValue(aux)
            , samplePerPixel(samplePerPixel)
        {}

        __device__ __host__ HYBRID_INLINE
        void operator()(RayAuxPPG& gOutPPG,
                        // Input    
                        const RayReg& ray,
                        // Index
                        uint16_t medumIndex,
                        const uint32_t localPixelId,
                        const uint32_t pixelSampleId) const
        {
            RayAuxPPG init = defaultValue;
            init.pixelIndex = localPixelId;
            init.pathIndex = localPixelId * samplePerPixel;
            gOutPPG = init;
        }
};