#pragma once


#include "RayLib/Vector.h"
#include "RayLib/Quaternion.h"

#include "Random.cuh"
#include "GPUPrimitiveP.cuh"
#include "DefaultLeaf.h"
#include "GPUTransformI.h"
#include "GPUSurface.h"
#include "DeviceMemory.h"
#include "TypeTraits.h"

#include "RayLib/CoordinateConversion.h"

// Triangle Memory Layout
struct SSphereData
{
    const float* radius;
};

using SSphereHit = Vector2f;

struct SSphereFunctions
{
    __device__ __forceinline__
    static Vector3f UVToNormal(const Vector2f& sphrCoords)
    {
        // Spherical to Cartesian
        Vector2f thetaPhi = Vector2f(// [-pi, pi]
                                     (sphrCoords[0] * MathConstants::Pi * 2.0f) - MathConstants::Pi,
                                      // [0, pi]
                                     (1.0f - sphrCoords[1]) * MathConstants::Pi);
        Vector3 dirZUp = Utility::SphericalToCartesianUnit(thetaPhi);
        // Spherical Coords calculates as Z up change it to Y up
        Vector3 dirYUp = Vector3(dirZUp[1], dirZUp[2], dirZUp[0]);
        return dirYUp;
    }

    __device__ __forceinline__
    static Vector2f CartesianToUV(const Vector3f& dir)
    {
       // Convert to spherical coordinates            
        Vector3 dirYUp = dir;
        Vector3 dirZup = -Vector3(dirYUp[2], dirYUp[0], dirYUp[1]);
        Vector2 thetaPhi = Utility::CartesianToSphericalUnit(dirZup);
        // Normalize to generate UV [0, 1]
        // tetha range [-pi, pi]
        float u = (thetaPhi[0] + MathConstants::Pi) * 0.5f / MathConstants::Pi;
        // If we are at edge point (u == 1) make it zero since 
        // piecewise constant function will not have that pdf (out of bounds)
        u = (u == 1.0f) ? 0.0f : u;
        // phi range [0, pi]
        float v = 1.0f - (thetaPhi[1] / MathConstants::Pi);
        // If (v == 1) then again pdf of would be out of bounds.
        // make it inbound
        v = (v == 1.0f) ? (v - MathConstants::SmallEpsilon) : v;
        return Vector2f{u, v};
    }

    __device__ __forceinline__
    static Vector3f Sample(// Output
                           Vector3f& normal,
                           float& pdf,
                           // Input
                           PrimitiveId primitiveId,
                           const SSphereData& primData,
                           // I-O
                           RandomGPU& rng)
    {
        // TODO: Implement
        return Zero3f;
    }

    __device__ __forceinline__
    static void PDF(// Outputs
                    Vector3f& normal,
                    float& pdf,
                    float& distance,
                    // Inputs
                    const Vector3f& position,
                    const Vector3f& direction,
                    const GPUTransformI& transform,
                    const PrimitiveId primitiveId,
                    const SSphereData& primData)
    {
        pdf = 1.0f;
        distance = primData.radius[primitiveId];
        normal = -direction;
    }

    // Triangle Hit Acceptance
    __device__ __forceinline__
    static HitResult Hit(// Output
                         HitKey& newMat,
                         PrimitiveId& newPrim,
                         SSphereHit& newHit,
                         // I-O
                         RayReg& rayData,
                         // Input
                         const GPUTransformI& transform,
                         const DefaultLeaf& leaf,
                         const SSphereData& primData)
    {
        // No need to transform the ray to local space
        // since t values does not vary between spaces
        float distance = primData.radius[leaf.primitiveId];        
        bool closerHit = (rayData.tMax <= distance);
        if(closerHit)
        {
            rayData.tMax = distance;
            newMat = leaf.matId;
            newPrim = leaf.primitiveId;

            newHit = CartesianToUV(transform.WorldToLocal(rayData.ray.getDirection(), true));
        }
        return HitResult{false, closerHit};
    }

    __device__ __forceinline__
    static AABB3f AABB(const GPUTransformI& transform,
                       //
                       PrimitiveId primitiveId,
                       const SSphereData& primData)
    {
        float distance = primData.radius[primitiveId];
        return AABB3f(Vector3f(-distance), Vector3f(distance));
    }

    __device__ __forceinline__
    static float Area(PrimitiveId primitiveId, const SSphereData& primData)
    {
        // Projected area
        float distance = primData.radius[primitiveId];
        return MathConstants::Pi * distance * distance;
    }

    __device__ __forceinline__
    static Vector3 Center(const GPUTransformI& transform,
                          PrimitiveId primitiveId, const SSphereData& primData)
    {        
        return Zero3;
    }

    static constexpr auto Leaf = GenerateDefaultLeaf<SSphereData>;
};

struct SSphereSurfaceGenerator
{
    __device__ __forceinline__
    static BasicSurface GenBasicSurface(const SSphereHit& sphrCoords,
                                        const GPUTransformI& transform,
                                        //
                                        PrimitiveId primitiveId,
                                        const SSphereData& primData)
    {
        Vector3f normal = SSphereFunctions::UVToNormal(sphrCoords);

        // Align this normal to Z axis to define tangent space rotation
        QuatF tbn = Quat::RotationBetweenZAxis(normal).Conjugate();
        tbn = tbn * transform.ToLocalRotation();

        return BasicSurface{tbn};
    }

    __device__ __forceinline__
    static SphrSurface GenSphrSurface(const SSphereHit& sphrCoords,
                                      const GPUTransformI& transform,
                                      //
                                      PrimitiveId primitiveId,
                                      const SSphereData& primData)
    {
        return SphrSurface{sphrCoords};
    }

    __device__ __forceinline__
    static UVSurface GenUVSurface(const SSphereHit& sphrCoords,
                                  const GPUTransformI& transform,
                                  //
                                  PrimitiveId primitiveId,
                                  const SSphereData& primData)
    {
        BasicSurface bs = GenBasicSurface(sphrCoords, transform,
                                          primitiveId, primData);
        return UVSurface{bs.worldToTangent, sphrCoords};
    }

    template <class Surface, SurfaceFunc<Surface, SSphereHit, SSphereData> SF>
    struct SurfaceFunctionType
    {
        using type = Surface;
        static constexpr auto SurfaceGeneratorFunction = SF;
    };

    static constexpr auto GeneratorFunctionList =
        std::make_tuple(SurfaceFunctionType<EmptySurface, 
                                            GenEmptySurface<SSphereHit, SSphereData>>{});

    template<class Surface>
    static constexpr SurfaceFunc<Surface, SSphereHit, SSphereData> GetSurfaceFunction()
    {
        using namespace PrimitiveSurfaceFind;
        return LoopAndFindType<Surface, SurfaceFunc<Surface, SSphereHit, SSphereData>,
                               decltype(GeneratorFunctionList)>(std::move(GeneratorFunctionList));
    }
};

class GPUPrimitiveSkySphere final
    : public GPUPrimitiveGroup<SSphereHit, SSphereData, DefaultLeaf,
                               SSphereSurfaceGenerator,
                               SSphereFunctions::Hit,
                               SSphereFunctions::Leaf, 
                               SSphereFunctions::AABB,
                               SSphereFunctions::Area, 
                               SSphereFunctions::Center,
                               SSphereFunctions::Sample,
                               SSphereFunctions::PDF>
{
    public:
        static constexpr const char*    TypeName() { return "Directional"; }

        static constexpr const char*    NAME_DISTANCE = "distance";

    private:
        DeviceMemory                            memory;
        // List of ranges for each batch
        uint64_t                                totalPrimitiveCount;
        // Misc Data
        std::map<uint32_t, Vector2ul>           batchRanges;
        std::map<uint32_t, AABB3>               batchAABBs;

    protected:
    public:
        // Constructors & Destructor
                                                GPUPrimitiveSkySphere();
                                                ~GPUPrimitiveSkySphere() = default;

        // Interface
        // Pirmitive type is used for delegating scene info to this class
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

        // Primitive Transform Info for accelerator
        PrimTransformType                       TransformType() const override;

        // Error check
        // Queries in order to check if this primitive group supports certain primitive data
        // Material may need that data
        bool                                    CanGenerateData(const std::string& s) const override;
};

static_assert(IsTracerClass<GPUPrimitiveSkySphere>::value,
              "GPUPrimitiveSkySphere is not a Tracer Class.");