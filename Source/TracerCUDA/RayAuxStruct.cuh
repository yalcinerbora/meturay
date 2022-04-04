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
    uint32_t        endpointIndex;  // Destination of the ray if applicable (i.e. NEE Ray)
    uint16_t        mediumIndex;    // Current Medium of the Ray
    uint8_t         depth;          // Current path depth
    RayType         type;           // Ray Type
    float           prevPDF;        // Previous Intersections BxDF pdf
                                    // (is used when a path ray hits a light (MIS))
};

struct RayAuxPPG
{
    // Path throughput
    // (a.k.a. total radiance coefficient along the path)
    Vector3f        radianceFactor;

    uint32_t        pixelIndex;     // Starting pixel index of the ray
    uint32_t        endpointIndex;  // Destination of the ray if applicable (i.e. NEE Ray)
    uint16_t        mediumIndex;    // Current Medium of the Ray
    uint8_t         depth;          // Current path depth
    RayType         type;           // Ray Type
    float           prevPDF;        // Previous Intersections BxDF pdf
                                    // (is used when a path ray hits a light (MIS))
    uint32_t        pathIndex;      // Global path node index
};

struct RayAuxRL
{
    // Path throughput
    // (a.k.a. total radiance coefficient along the path)
    Vector3f    radianceFactor;

    uint32_t    pixelIndex;         // Starting pixel index of the ray
    uint32_t    endpointIndex;      // Destination of the ray if applicable (i.e. NEE Ray)
    uint16_t    mediumIndex;        // Current Medium of the Ray
    uint8_t     depth;              // Current path depth
    RayType     type;               // Ray Type
    float       prevPDF;            // Previous Intersections BxDF pdf
                                    // (is used when a path ray hits a light (MIS))
    uint32_t    prevSpatialIndex;   // Previous Spatial Index (QFunction) in the path chain
    float       prevLumReflectance; // Previous reflectance factor of the ray.
};

struct RayAuxWFPG
{
    // Path throughput
    // (a.k.a. total radiance coefficient along the path)
    Vector3f        radianceFactor;

    uint32_t        pixelIndex;     // Starting pixel index of the ray
    uint32_t        endpointIndex;  // Destination of the ray if applicable (i.e. NEE Ray)
    uint16_t        mediumIndex;    // Current Medium of the Ray
    uint8_t         depth;          // Current path depth
    RayType         type;           // Ray Type
    float           prevPDF;        // Previous Intersections BxDF pdf
                                    // (is used when a path ray hits a light (MIS))
    // Method Related
    uint32_t        pathIndex;      // Index of the path in the path array
    uint32_t        binId;          // Position bin id
    Vector2h        guideDir;       // Spherical coords of the guided direction
    half            guidePDF;       // Guided direction's sample pdf
};

static const RayAuxBasic InitialBasicAux = RayAuxBasic
{
    UINT32_MAX
};

static const RayAuxPath InitialPathAux = RayAuxPath
{
    Vector3f(1.0f, 1.0f, 1.0f),
    UINT32_MAX, UINT32_MAX, UINT16_MAX,
    0,  // Depth
    RayType::CAMERA_RAY,
    NAN
};

static const RayAuxAO InitialAOAux = RayAuxAO
{
    Vector3f(1.0f, 1.0f, 1.0f),
    UINT32_MAX
};

static const RayAuxPPG InitialPPGAux = RayAuxPPG
{
    Vector3f(1.0f, 1.0f, 1.0f),
    UINT32_MAX, UINT32_MAX, UINT16_MAX,
    0,  // Depth
    RayType::CAMERA_RAY,
    NAN,
    UINT32_MAX
};

static const RayAuxRL InitialRLAux = RayAuxRL
{
    Vector3f(1.0f, 1.0f, 1.0f),
    UINT32_MAX, UINT32_MAX, UINT16_MAX,
    0,  // Depth
    RayType::CAMERA_RAY,
    NAN,
    UINT32_MAX,
    NAN
};

static const RayAuxWFPG InitialWFPGAux = RayAuxWFPG
{
    Vector3f(1.0f, 1.0f, 1.0f),
    UINT32_MAX, UINT32_MAX, UINT16_MAX,
    0,
    RayType::CAMERA_RAY,
    NAN,
    // WFPG Related
    UINT32_MAX,
    UINT32_MAX,
    Vector2h(0.0f, 0.0f),
    0.0f
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
                        const RayReg&,
                        // Index
                        uint16_t,
                        const uint32_t localPixelId,
                        const uint32_t) const
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
                        const RayReg&,
                        // Index
                        uint16_t mediumIndex,
                        const uint32_t localPixelId,
                        const uint32_t) const
        {
            RayAuxPath init = defaultValue;
            init.pixelIndex = localPixelId;
            init.type = RayType::CAMERA_RAY;
            init.mediumIndex = mediumIndex;
            init.depth = 1;
            gOutPath = init;
        }
};

class RayAuxInitRefPG
{
     private:
        RayAuxPath defaultValue;

    public:
        RayAuxInitRefPG(const RayAuxPath& aux)
            : defaultValue(aux)
        {}

        __device__ __host__ HYBRID_INLINE
        void operator()(RayAuxPath& gOutPath,
                        // Input
                        const RayReg&,
                        // Index
                        uint16_t mediumIndex,
                        const uint32_t,
                        const uint32_t) const
        {
            RayAuxPath init = defaultValue;
            init.pixelIndex = UINT32_MAX;
            init.type = RayType::CAMERA_RAY;
            init.mediumIndex = mediumIndex;
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
                        const RayReg&,
                        // Index
                        uint16_t,
                        const uint32_t localPixelId,
                        const uint32_t) const
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
                        const RayReg&,
                        // Index
                        uint16_t medumIndex,
                        const uint32_t localPixelId,
                        const uint32_t pixelSampleId) const
        {
            RayAuxPPG init = defaultValue;
            init.pixelIndex = localPixelId;
            init.pathIndex = localPixelId * samplePerPixel + pixelSampleId;
            init.depth = 1;
            init.mediumIndex = medumIndex;
            gOutPPG = init;
        }
};

class RayAuxInitRL
{
    private:
        RayAuxRL    defaultValue;
        uint32_t    samplePerPixel;

    public:
        RayAuxInitRL(const RayAuxRL& aux,
                     uint32_t samplePerPixel)
            : defaultValue(aux)
            , samplePerPixel(samplePerPixel)
        {}

        __device__ __host__ HYBRID_INLINE
        void operator()(RayAuxRL& gOutPPG,
                        // Input
                        const RayReg&,
                        // Index
                        uint16_t medumIndex,
                        const uint32_t localPixelId,
                        const uint32_t pixelSampleId) const
        {
            RayAuxRL init = defaultValue;
            init.pixelIndex = localPixelId;
            init.depth = 1;
            init.mediumIndex = medumIndex;
            gOutPPG = init;
        }
};

class RayAuxInitWFPG
{
    private:
        RayAuxWFPG  defaultValue;
        uint32_t    samplePerPixel;

    public:
        RayAuxInitWFPG(const RayAuxWFPG& aux,
                      uint32_t samplePerPixel)
            : defaultValue(aux)
            , samplePerPixel(samplePerPixel)
        {}

        __device__ __host__ HYBRID_INLINE
        void operator()(RayAuxWFPG& gOutPPG,
                        // Input
                        const RayReg&,
                        // Index
                        uint16_t medumIndex,
                        const uint32_t localPixelId,
                        const uint32_t pixelSampleId) const
        {
            RayAuxWFPG init = defaultValue;
            init.pixelIndex = localPixelId;
            init.pathIndex = localPixelId * samplePerPixel + pixelSampleId;
            init.depth = 1;
            init.mediumIndex = medumIndex;
            gOutPPG = init;
        }
};