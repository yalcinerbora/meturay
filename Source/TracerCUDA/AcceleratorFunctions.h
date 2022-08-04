#pragma once

#include "RayStructs.h"
#include "GPUTransformI.h"

#include "RayLib/HitStructs.h"
#include "RayLib/AABB.h"

using HitKeyList = std::array<HitKey, SceneConstants::MaxPrimitivePerSurface>;
using PrimitiveRangeList = std::array<Vector2ul, SceneConstants::MaxPrimitivePerSurface>;
using PrimitiveIdList = std::array<uint32_t, SceneConstants::MaxPrimitivePerSurface>;

using HitResult = Vector<2, bool>;

class GPUTransformI;
class RNGeneratorGPUI;

// This is Leaf of Base Accelerator
// It points to another accelerator pair
struct /*alignas(16)*/ BaseLeaf
{
    Vector3f    aabbMin;
    HitKey      accKey;
    Vector3f    aabbMax;
};

template <class PrimitiveData, class LeafData>
using LeafGenFunction = LeafData(*)(const HitKey matId,
                                    const PrimitiveId primitiveId,
                                    const PrimitiveData& primData);

// Custom bounding box generation function
// For primitive
template <class PrimitiveData>
using BoxGenFunction = AABB3f(*)(const GPUTransformI&,
                                 PrimitiveId primitiveId,
                                 const PrimitiveData&);

// Surface area generation function for bound hierarchy generation
template <class PrimitiveData>
using AreaGenFunction = float(*)(const GPUTransformI& transform,
                                 PrimitiveId primitiveId, const PrimitiveData&);

// Center generation function for bound hierarchy generation
template <class PrimitiveData>
using CenterGenFunction = Vector3(*)(const GPUTransformI& transform,
                                     PrimitiveId primitiveId, const PrimitiveData&);

// Sample function for generating rays and/or finding an arbitrary point
template <class PrimitiveData>
using SamplePosFunction = Vector3(*)(Vector3& normal,
                                     float& pdf,
                                     // Inputs,
                                     const GPUTransformI&,
                                     //
                                     PrimitiveId primitiveId,
                                     const PrimitiveData&,
                                     // I-O
                                     RNGeneratorGPUI& rng);

// PDF function for calculating a PDF function for hitting from such
// position/direction
template <class PrimitiveData>
using PDFPosRefFunction = void(*)(// Outputs
                                  Vector3f& normal,
                                  float& pdf,
                                  float& distance,
                                  // Inputs
                                  const RayF& ray,
                                  const GPUTransformI& transform,
                                  //
                                  const PrimitiveId primitiveId,
                                  const PrimitiveData& primData);

template <class PrimitiveData>
using PDFPosHitFunction = float(*)(// Inputs
                                   const Vector3f& hitPosition,
                                   const Vector3f& hitDirection,
                                   const QuatF& tbnRotation,
                                   const GPUTransformI& transform,
                                   //
                                   const PrimitiveId primitiveId,
                                   const PrimitiveData& primData);

template <class PrimitiveData, uint32_t PosCount>
using AcquirePosFunction = void(*)(// Output
                                   Vector3f positions[PosCount],
                                   // Inputs
                                   PrimitiveId primitiveId,
                                   const PrimitiveData& primData);

template <class HitData, class PrimitiveData, class LeafData>
using IntersectsFunction = bool(*)(// Output
                                   float& newT,
                                   HitData& newHit,
                                   // I-O
                                   const RayReg& rayData,
                                   // Input
                                   const GPUTransformI& transform,
                                   const LeafData& leaf,
                                   const PrimitiveData& primData);

template <class HitData, class PrimitiveData, class LeafData>
using AlphaTestFunction = bool(*)(// Input
                                  const HitData& potentialHit,
                                  const LeafData& leaf,
                                  const PrimitiveData& primData);

template <class PrimitiveData>
using VoxelizeFunction = uint32_t(*)(uint64_t* gVoxMortonCodes,
                                     Vector2us* gVoxelNormals,
                                     uint32_t voxIndMaxCount,
                                     // Inputs
                                     bool onlyCalcSize,
                                     PrimitiveId primitiveId,
                                     const PrimitiveData& primData,
                                     const GPUTransformI& transform,
                                     // Voxel Inputs
                                     const AABB3f& sceneAABB,
                                     uint32_t resolutionXYZ);

// Common Functors for gpu AABB Generation
template<class PrimitiveGroup>
struct AABBGen
{
    public:
        using PData = typename PrimitiveGroup::PrimitiveData;

    private:

        PData                   pData;
        const GPUTransformI&    transform;

    protected:
    public:
        // Constructors & Destructor
        AABBGen(PData pData, const GPUTransformI& t)
            : pData(pData)
            , transform(t)
        {}

        __device__ __host__
        inline AABB3f operator()(const PrimitiveId& id) const
        {
            return PrimitiveGroup::AABB(transform, id, pData);
        }
};

template<class PrimitiveGroup>
struct CentroidGen
{
    public:
        using PData = typename PrimitiveGroup::PrimitiveData;

    private:
        PData                   pData;
        const GPUTransformI&    transform;

    protected:
    public:
        // Constructors & Destructor
        CentroidGen(PData pData, const GPUTransformI& t)
            : pData(pData)
            , transform(t)
        {}

        __device__ __host__ HYBRID_INLINE
        Vector3 operator()(const PrimitiveId& id) const
        {
            return PrimitiveGroup::Center(transform, id, pData);
        }
};

struct AABBUnion
{
    __device__ __host__ HYBRID_INLINE
    AABB3f operator()(const AABB3f& a,
                      const AABB3f& b) const
    {
        return a.Union(b);
    }
};

// Defaults (Empty Implementations of above classes)
// Custom bounding box generation function
// For primitive
template <class PrimitiveData>
__device__ inline
AABB3f DefaultAABBGen(const GPUTransformI&,
                      PrimitiveId,
                      const PrimitiveData&)
{
    Vector3f minInf(-INFINITY);
    return AABB3f(minInf, minInf);
}

// Surface area generation function for bound hierarchy generation
template <class PrimitiveData>
__device__ inline
float DefaultAreaGen(const GPUTransformI&, PrimitiveId, const PrimitiveData&)
{
    return 0.0f;
}

// Center generation function for bound hierarchy generation
template <class PrimitiveData>
__device__ inline
Vector3 DefaultCenterGen(const GPUTransformI&,
                         PrimitiveId, const PrimitiveData&)
{
    return Vector3f(INFINITY);
}

template <class PrimitiveData>
__device__ inline
Vector3 DefaultSamplePos(Vector3& normal, float& pdf,
                         // Input
                         const GPUTransformI&,
                         //
                         PrimitiveId,
                         const PrimitiveData&,
                         // I-O
                         RNGeneratorGPUI&)
{
    normal = Vector3f(INFINITY);
    pdf = 0.0f;
    return Vector3f(INFINITY);
}

template <class PrimitiveData>
__device__ inline
void DefaultPDFPosRef(// Outputs
                   Vector3f& normal,
                   float& pdf,
                   float& distance,
                   // Inputs
                   const RayF&,
                   const GPUTransformI&,
                   const PrimitiveId,
                   const PrimitiveData&)
{
    normal = Vector3f(INFINITY);
    pdf = 0.0f;
    distance = INFINITY;
}

template <class PrimitiveData>
__device__ inline
float DefaultPDFPosHit(// Inputs
                       const Vector3f&,
                       const Vector3f&,
                       const QuatF&,
                       const GPUTransformI&,
                       //
                       const PrimitiveId,
                       const PrimitiveData&)
{
    return 0.0f;
}

template <class PrimitiveData, uint32_t PosCount>
__device__ inline
void DefaultAcqPosition(// Output
                         Vector3f positions[PosCount],
                         // Inputs
                         PrimitiveId,
                         const PrimitiveData&)
{
    positions[0] = Zero3f;
}

template <class HitData, class PrimitiveData, class LeafData>
__device__ inline
bool DefaultAlphaTest(const HitData&,
                           const LeafData&,
                           const PrimitiveData&)
{
    return true;
}

template <class HitData, class PrimitiveData, class LeafData>
__device__ inline
bool DefaultIntersects(// Output
                       float&,
                       HitData&,
                       // I-O
                       const RayReg&,
                       // Input
                       const GPUTransformI&,
                       const LeafData&,
                       const PrimitiveData&)
{
    // Always no Intersection
    return false;
}

template <class PrimitiveData>
__device__ inline
static uint32_t DefaultVoxelize(uint64_t*,
                                Vector2us*,
                                uint32_t,
                                // Inputs
                                bool,
                                PrimitiveId,
                                const PrimitiveData&,
                                const GPUTransformI&,
                                // Voxel Inputs
                                const AABB3f&,
                                uint32_t)
{
    return 0;
}
