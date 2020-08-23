#pragma once
/**

Default Sphere Implementation
One of the fundamental functional types.

Has two types of data
Position and radius.

All of them should be provided

*/

#include <map>
#include <cuda_fp16.h>

#include "DefaultLeaf.h"
#include "GPUPrimitiveP.cuh"
#include "DeviceMemory.h"
#include "TypeTraits.h"
#include "GPUSurface.h"

#include "RayLib/PrimitiveDataTypes.h"
#include "RayLib/Vector.h"
#include "RayLib/Sphere.h"

// Sphere memory layout
struct SphereData
{
    const Vector4f* centerRadius;
};

// Hit of sphere is spherical coordinates
using SphereHit = Vector2f;


struct SphrFunctions
{
    // Sphere Hit Acceptance
    __device__ __host__
    static inline HitResult Hit(// Output
                                HitKey& newMat,
                                PrimitiveId& newPrimitive,
                                SphereHit& newHit,
                                // I-O
                                RayReg& rayData,
                                // Input
                                const GPUTransformI& transform,
                                const DefaultLeaf& leaf,
                                const SphereData& primData)
    {
        // Get Packed data and unpack
        Vector4f data = primData.centerRadius[leaf.primitiveId];
        Vector3f center = data;
        float radius = data[3];

        // Do Intersecton test on local space
        RayF r = transform.WorldToLocal(rayData.ray);
        Vector3 pos; float newT;
        bool intersects = r.IntersectsSphere(pos, newT, center, radius);

        // Check if the hit is closer
        bool closerHit = intersects && (newT < rayData.tMax);
        if(closerHit)
        {
            rayData.tMax = newT;
            newMat = leaf.matId;
            newPrimitive = leaf.primitiveId;

            // Gen Spherical Coords (R can be fetched using primitiveId)
            // Clamp acos input for singularity
            Vector3 relativeCoord = pos - center;
            float tethaCos = HybridFuncs::Clamp(relativeCoord[2] / radius, -1.0f, 1.0f);
            float tetha = acos(tethaCos);
            float phi = atan2(relativeCoord[1], relativeCoord[0]);
            newHit = Vector2(tetha, phi);
        }
        return HitResult{false, closerHit};
    }

    __device__ __host__
    static inline AABB3f AABB(const GPUTransformI& transform, 
                              PrimitiveId primitiveId,
                              const SphereData& primData)
    {
        // Get Packed data and unpack
        Vector4f data = primData.centerRadius[primitiveId];
        Vector3f center = data;
        float radius = data[3];

        return Sphere::BoundingBox(center, radius);
    }

    __device__ __host__
    static inline float Area(PrimitiveId primitiveId,
                             const SphereData& primData)
    {
        Vector4f data = primData.centerRadius[primitiveId];
        float radius = data[3];

        // Surface area is related to radius only (wrt of its square)
        // TODO: check if this is a good estimation
        return radius * radius;
    }

    __device__ __host__
    static inline Vector3 Center(PrimitiveId primitiveId,
                                 const SphereData& primData)
    {
        Vector4f data = primData.centerRadius[primitiveId];
        Vector3f center = data;

        return center;
    }

    static constexpr auto Leaf          = GenerateDefaultLeaf<SphereData>;    
};

struct SphereSurfaceGenerator
{
    __device__ __host__
    static inline BasicSurface GenBasicSurface(const SphereHit& sphrCoords,
                                               const GPUTransformI& transform,
                                               //
                                               PrimitiveId primitiveId,
                                               const SphereData& primData)
    {
        Vector4f data = primData.centerRadius[primitiveId];
        Vector3f center = data;
        float radius = data[3];

        // Convert spherical hit to cartesian
        Vector3 normal = Vector3(sin(sphrCoords[0]) * cos(sphrCoords[1]),
                                 sin(sphrCoords[0]) * sin(sphrCoords[1]),
                                 cos(sphrCoords[0]));

        // Align this normal to Z axis to define tangent space rotation
        QuatF tbn = Quat::RotationBetweenZAxis(normal);
        tbn = tbn * transform.ToLocalRotation();

        return BasicSurface{tbn};
    }

    __device__ __host__
    static inline SphrSurface GenSphrSurface(const SphereHit& sphrCoords,
                                             const GPUTransformI& transform,
                                             //
                                             PrimitiveId primitiveId,
                                             const SphereData& primData)
    {        
        return SphrSurface{sphrCoords};
    }

    __device__ __host__
    static inline UVSurface GenUVSurface(const SphereHit& sphrCoords,
                                         const GPUTransformI& transform,
                                         //
                                         PrimitiveId primitiveId,
                                         const SphereData& primData)
    {
        BasicSurface bs = GenBasicSurface(sphrCoords, transform,
                                          primitiveId, primData);

        // Gen UV    
        Vector2 uv = sphrCoords;
        // tetha is [-pi, pi], normalize
        uv[0] = (uv[0] + MathConstants::Pi) * 0.5f * MathConstants::InvPi;
        // phi is [0, pi], normalize 
        uv[1] /= MathConstants::Pi;

        return UVSurface{bs.worldToTangent, uv};
    }


    template <class Surface, SurfaceFunc<Surface, SphereHit, SphereData> SF>
    struct SurfaceFunctionType
    {
        using type = Surface;
        static constexpr auto SurfaceGeneratorFunction = SF;
    };

    static constexpr auto GeneratorFunctionList = 
        std::make_tuple(SurfaceFunctionType<EmptySurface, GenEmptySurface<SphereHit, SphereData>>{},
                        SurfaceFunctionType<BasicSurface, GenBasicSurface>{},
                        SurfaceFunctionType<SphrSurface, GenSphrSurface>{},
                        SurfaceFunctionType<UVSurface, GenUVSurface>{});

    template<class Surface>
    static constexpr SurfaceFunc<Surface, SphereHit, SphereData> GetSurfaceFunction()
    {
        using namespace PrimitiveSurfaceFind;
        return LoopAndFindType<Surface, SurfaceFunc<Surface, SphereHit, SphereData>,
                               decltype(GeneratorFunctionList)>(std::move(GeneratorFunctionList));
    }
};

class GPUPrimitiveSphere final
    : public GPUPrimitiveGroup<SphereHit, SphereData, DefaultLeaf,
                               SphereSurfaceGenerator, SphrFunctions::Hit,
                               SphrFunctions::Leaf, SphrFunctions::AABB, 
                               SphrFunctions::Area, SphrFunctions::Center>
{
    public:
        static constexpr const char*            TypeName() { return "Sphere"; }

        static constexpr PrimitiveDataLayout    POS_LAYOUT = PrimitiveDataLayout::FLOAT_3;
        static constexpr PrimitiveDataLayout    RADUIS_LAYOUT = PrimitiveDataLayout::FLOAT_1;

    private:
        DeviceMemory                            memory;

        // List of ranges for each batch
        uint64_t                                totalPrimitiveCount;
        std::map<uint32_t, Vector2ul>           batchRanges;
        std::map<uint32_t, AABB3>               batchAABBs;

    public:
        // Constructors & Destructor
                                                GPUPrimitiveSphere();
                                                ~GPUPrimitiveSphere() = default;

        // Interface
        // Pirmitive type is used for delegating scene info to this class
        const char*                             Type() const override;
        // Allocates and Generates Data
        SceneError                              InitializeGroup(const NodeListing& surfaceDatalNodes, double time,
                                                                const SurfaceLoaderGeneratorI& loaderGen,
                                                                const std::string& scenePath) override;
        SceneError                              ChangeTime(const NodeListing& surfaceDatalNodes, double time,
                                                           const SurfaceLoaderGeneratorI& loaderGen,
                                                           const std::string& scenePath) override;
        // Provides data to Event Estimator
        bool                                    HasPrimitive(uint32_t surfaceDataId) const override;
        SceneError                              GenerateLights(std::vector<CPULight>&,
                                                               const GPUDistribution2D&, HitKey key,
                                                               uint32_t surfaceDataId,
                                                               const Matrix4x4& transform) const override;
        // Access primitive range from Id
        Vector2ul                               PrimitiveBatchRange(uint32_t surfaceDataId) const override;
        AABB3                                   PrimitiveBatchAABB(uint32_t surfaceDataId) const override;

        // Error check
        // Queries in order to check if this primitive group supports certain primitive data
        // Material may need that data
        bool                                    CanGenerateData(const std::string& s) const override;
};

static_assert(IsTracerClass<GPUPrimitiveSphere>::value,
              "GPUPrimitiveSphere is not a Tracer Class.");