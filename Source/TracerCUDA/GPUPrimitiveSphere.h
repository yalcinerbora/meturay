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
#include "RNGenerator.h"
#include "GPUPrimitiveP.cuh"
#include "DeviceMemory.h"
#include "TypeTraits.h"
#include "GPUSurface.h"

#include "RayLib/PrimitiveDataTypes.h"
#include "RayLib/Vector.h"
#include "RayLib/Sphere.h"
#include "RayLib/CoordinateConversion.h"

// Sphere memory layout
struct SphereData
{
    const Vector4f* centerRadius;
};

// Hit of sphere is spherical coordinates
using SphereHit = Vector2f;

struct SphrFunctions
{
    __device__ inline
    static Vector3f SamplePosition(// Output
                                   Vector3f& normal,
                                   float& pdf,
                                   // Input
                                   const GPUTransformI& transform,
                                   //
                                   PrimitiveId primitiveId,
                                   const SphereData& primData,
                                   // I-O
                                   RNGeneratorGPUI& rng)
    {
        Vector4f data = primData.centerRadius[primitiveId];
        Vector3f center = data;
        float radius = data[3];

        // http://mathworld.wolfram.com/SpherePointPicking.html

        Vector2f xi = rng.Uniform2D();

        float theta = 2.0f * MathConstants::Pi * xi[0];
        float cosPhi = 2.0f * xi[1] - 1.0f;
        float sinPhi = sqrtf(fmaxf(0.0f, 1.0f - cosPhi * cosPhi));

        Vector3f unitPos = Utility::SphericalToCartesianUnit(Vector2f(sin(theta), cos(theta)),
                                                             Vector2f(sinPhi, cosPhi));

        // Calculate PDF
        // Approximate the area with the determinant
        float area = SphrFunctions::Area(transform, primitiveId, primData);
        pdf = 1.0f / area;

        Vector3f sphrLoc = center + radius * unitPos;
        normal = unitPos;

        sphrLoc = transform.LocalToWorld(sphrLoc);
        normal = transform.LocalToWorld(normal, true);

        return sphrLoc;
    }

    __device__ inline
    static void PositionPdfFromReference(// Outputs
                                         Vector3f& normal,
                                         float& pdf,
                                         float& distance,
                                         // Inputs
                                         const RayF& ray,
                                         const GPUTransformI& transform,
                                         //
                                         const PrimitiveId primitiveId,
                                         const SphereData& primData)
    {
        Vector4f data = primData.centerRadius[primitiveId];
        Vector3f center = data;
        float radius = data[3];

        RayF r = ray;
        r = transform.WorldToLocal(r);

        Vector3 sphrPos;
        bool intersects = r.IntersectsSphere(sphrPos, distance,
                                             center, radius);

        sphrPos = transform.LocalToWorld(sphrPos);
        normal = (ray.getPosition() - sphrPos).Normalize();

        // Return non zero if it intersected
        if(intersects)
        {
            // Approximate the area with the determinant
            float area = SphrFunctions::Area(transform, primitiveId, primData);
            pdf = 1.0f / area;
        }
        else pdf = 0.0f;
    }

    __device__ inline
    static float PositionPdfFromHit(// Inputs
                                    const Vector3f&,
                                    const Vector3f&,
                                    const QuatF&,
                                    const GPUTransformI& transform,
                                    //
                                    const PrimitiveId primitiveId,
                                    const SphereData& primData)
    {
        // Approximate the area with the determinant
        float area = SphrFunctions::Area(transform, primitiveId, primData);
        return 1.0f / area;
    }

    template <class GPUTransform>
    __device__ inline
    static bool IntersectsT(// Output
                            float& newT,
                            SphereHit& newHit,
                            // I-O
                            const RayReg& rayData,
                            // Input
                            const GPUTransform& transform,
                            const DefaultLeaf& leaf,
                            const SphereData& primData)
    {
        // Get Packed data and unpack
        const Vector4f& data = primData.centerRadius[leaf.primitiveId];
        Vector3f center = data;
        float radius = data[3];

        // Do Intersection test on local space
        RayF r = transform.WorldToLocal(rayData.ray);
        Vector3 pos; float t;
        bool intersects = r.IntersectsSphere(pos, t, center, radius);

        if(intersects)
        {
            newT = t;
            // Gen Spherical Coords (R can be fetched using primitiveId later)
            // Clamp acos input for singularity
            Vector3 relativeCoord = pos - center;
            float tethaCos = HybridFuncs::Clamp(relativeCoord[2] / radius, -1.0f, 1.0f);
            float tetha = acos(tethaCos);
            float phi = atan2(relativeCoord[1], relativeCoord[0]);
            newHit = SphereHit(tetha, phi);
        }
        return intersects;
    }

    static constexpr auto& Intersects = IntersectsT<GPUTransformI>;

    // TODO: Implement Alpha test for Spheres
    static constexpr auto& AlphaTest = DefaultAlphaTest<SphereHit, SphereData, DefaultLeaf>;

    __device__ inline
    static AABB3f AABB(const GPUTransformI& transform,
                       PrimitiveId primitiveId,
                       const SphereData& primData)
    {
        // TODO: incorporate transform here
        // Get Packed data and unpack
        Vector4f data = primData.centerRadius[primitiveId];
        Vector3f center = data;
        float radius = data[3];

        return transform.LocalToWorld(Sphere::BoundingBox(center, radius));
    }

    __device__ inline
    static float Area(const GPUTransformI& transform,
                      PrimitiveId primitiveId,
                      const SphereData& primData)
    {
        Vector4f data = primData.centerRadius[primitiveId];
        float radius = data[3];

        // https://math.stackexchange.com/questions/942561/surface-area-of-transformed-sphere
        static constexpr float p = 8.0f / 5.0f;
        static constexpr float pRecip = 1.0f / p;

        Vector3f semiAxes = radius * transform.ToWorldScale();
        float approxArea = pow(semiAxes[1] * semiAxes[2], p);
        approxArea += pow(semiAxes[2] * semiAxes[0], p);
        approxArea += pow(semiAxes[0] * semiAxes[1], p);
        approxArea *= 0.3333f;
        approxArea = pow(approxArea, pRecip);
        approxArea *= 4.0f * MathConstants::Pi;
        return approxArea;
    }

    __device__ inline
    static Vector3 Center(const GPUTransformI& transform,
                          PrimitiveId primitiveId,
                          const SphereData& primData)
    {
        // TODO: incorporate transform here
        Vector4f data = primData.centerRadius[primitiveId];
        Vector3f center = data;

        return transform.LocalToWorld(center);
    }

    __device__ inline
    static void AcquirePositions(// Output
                                 Vector3f positions[1],
                                 // Inputs
                                 PrimitiveId primitiveId,
                                 const SphereData& primData)
    {
        positions[0] = primData.centerRadius[primitiveId];
    }

    static constexpr auto& Leaf = GenerateDefaultLeaf<SphereData>;
};

struct SphereSurfaceGenerator
{
    __device__ inline
    static BasicSurface GenBasicSurface(const SphereHit& sphrCoords,
                                        const GPUTransformI& transform,
                                        //
                                        PrimitiveId,
                                        const SphereData&)
    {
        // Convert spherical hit to cartesian
        Vector3 normal = Vector3(sin(sphrCoords[0]) * cos(sphrCoords[1]),
                                 sin(sphrCoords[0]) * sin(sphrCoords[1]),
                                 cos(sphrCoords[0]));

        // Align this normal to Z axis to define tangent space rotation
        QuatF tbn = Quat::RotationBetweenZAxis(normal).Conjugate();
        tbn = tbn * transform.ToLocalRotation();

        return BasicSurface{tbn};
    }

    __device__ inline
    static SphrSurface GenSphrSurface(const SphereHit& sphrCoords,
                                      const GPUTransformI&,
                                      //
                                      PrimitiveId,
                                      const SphereData&)
    {
        return SphrSurface{sphrCoords};
    }

    __device__ inline
    static UVSurface GenUVSurface(const SphereHit& sphrCoords,
                                  const GPUTransformI& transform,
                                  //
                                  PrimitiveId primitiveId,
                                  const SphereData& primData)
    {
        BasicSurface bs = GenBasicSurface(sphrCoords, transform,
                                          primitiveId, primData);

        // Gen UV
        Vector2 uv = sphrCoords;
        // theta is [0, 2 * pi], normalize
        uv[0] *= 0.5 * MathConstants::InvPi;
        // phi is [-pi/2, pi/2], normalize
        uv[1] = uv[1] * MathConstants::InvPi + 0.5f;

        return UVSurface{bs.worldToTangent, uv};
    }

    template <class Surface, SurfaceFunc<Surface, SphereHit, SphereData> SF>
    struct SurfaceFunctionType
    {
        using type = Surface;
        static constexpr auto SurfaceGeneratorFunction = SF;
    };

    static constexpr auto GeneratorFunctionList =
        std::make_tuple(SurfaceFunctionType<EmptySurface, DefaultGenEmptySurface<SphereHit, SphereData>>{},
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
                               SphereSurfaceGenerator, SphrFunctions,
                               PrimTransformType::CONSTANT_LOCAL_TRANSFORM>
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
        // Primitive type is used for delegating scene info to this class
        const char*                             Type() const override;
        // Allocates and Generates Data
        SceneError                              InitializeGroup(const NodeListing& surfaceDataNodes, double time,
                                                                const SurfaceLoaderGeneratorI& loaderGen,
                                                                const TextureNodeMap& textureNodes,
                                                                const std::string& scenePath) override;
        SceneError                              ChangeTime(const NodeListing& surfaceDatalNodes, double time,
                                                           const SurfaceLoaderGeneratorI& loaderGen,
                                                           const std::string& scenePath) override;
        // Access primitive range from Id
        Vector2ul                               PrimitiveBatchRange(uint32_t surfaceDataId) const override;
        AABB3                                   PrimitiveBatchAABB(uint32_t surfaceDataId) const override;
        bool                                    PrimitiveBatchHasAlphaMap(uint32_t surfaceDataId) const override;
        bool                                    PrimitiveBatchBackFaceCulled(uint32_t surfaceDataId) const override;
        // Query
        // How many primitives are available on this class
        // This includes the indexed primitive count
        uint64_t                                TotalPrimitiveCount() const override;
        // Total primitive count but not indexed
        uint64_t                                TotalDataCount() const override;

        // Error check
        // Queries in order to check if this primitive group supports certain primitive data
        // Material may need that data
        bool                                    CanGenerateData(const std::string& s) const override;
};

static_assert(IsTracerClass<GPUPrimitiveSphere>::value,
              "GPUPrimitiveSphere is not a Tracer Class.");