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
class RandomGPU;

// This is Leaf of Base Accelerator
// It points to another accelerator pair
struct /*alignas(16)*/ BaseLeaf
{
    Vector3f aabbMin;
    HitKey accKey;
    Vector3f aabbMax;
};

// Accept hit function
// Return two booleans first boolean tells control flow to terminate
// intersection checking (when finding any hit is enough),
// other boolean is returns that the hit is accepted or not.
//
// If hit is accepted Accept hit function should return
// valid material key, primitiveId, and hit.
//
// Material Key is the index of material
// Primitive id is the index of the individual primitive
// Hit is the interpolting weights of the primitive
//
// PrimitiveData struct holds the array of the primitive data
// (normal, position etc..)
template <class HitData, class PrimitiveData, class LeafData>
using AcceptHitFunction = HitResult(*)(// Output
                                       HitKey&,
                                       PrimitiveId&,
                                       HitData&,
                                       // I-O
                                       RayReg& r,
                                       // Input
                                       const GPUTransformI&,
                                       const LeafData& data,
                                       const PrimitiveData& gPrimData);

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

// Surface area generation function for bound hierarcy generation
template <class PrimitiveData>
using AreaGenFunction = float(*)(PrimitiveId primitiveId, const PrimitiveData&);

// Center generation function for bound hierarcy generation
template <class PrimitiveData>
using CenterGenFunction = Vector3(*)(PrimitiveId primitiveId, const PrimitiveData&);

// Sample function for generating rays and/or finding an arbitrary point
template <class PrimitiveData>
using SampleFunction = Vector3(*)(Vector3& normal,
                                  float& pdf,

                                  PrimitiveId primitiveId,
                                  const PrimitiveData&,
                                  // I-O
                                  RandomGPU& rng);


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
        __forceinline__ AABB3f operator()(const PrimitiveId& id) const
        {
            return PrimitiveGroup::AABB(transform, id, pData);
        }
};

struct AABBUnion
{
    __device__ __host__
    __forceinline__ AABB3f operator()(const AABB3f& a,
                                      const AABB3f& b) const
    {
        return a.Union(b);
    }
};