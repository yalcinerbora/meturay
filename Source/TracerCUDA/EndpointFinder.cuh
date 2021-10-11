#pragma once

#include "RayLib/HitStructs.h"

#include "BinarySearch.cuh"

class EndpointFinder
{
    private:       
        // First Binary Search Data
        const uint32_t          transformIdCount;
        const TransformId*      dTransformIds;
        const uint64_t*         dPrimIdRanges;
        // Second Binary Search Data
        const PrimitiveId*      dPrimitiveIds;
        const uint32_t*         dEndPointIndices;
    protected:
    public:
        // Constructors & Destructor
        __host__                EndpointFinder();
        __host__                EndpointFinder(uint32_t transformIdCount,
                                               TransformId* dTransformIds,
                                               uint64_t* dPrimIdRanges,
                                               PrimitiveId* dPrimitiveIds,
                                               uint32_t* dEndPointIndices);
                                ~EndpointFinder() = default;

        // Methods
        __device__ uint32_t     FindEndPointIndex(TransformId transformId,
                                                  PrimitiveId primitiveId) const;

};

__host__ inline
EndpointFinder::EndpointFinder()
    : transformIdCount(0)
    , dTransformIds(nullptr)
    , dPrimIdRanges(nullptr)
    , dPrimitiveIds(nullptr)
    , dEndPointIndices(nullptr)
{}

__host__ inline
EndpointFinder::EndpointFinder(uint32_t transformIdCount,
                               TransformId* dTransformIds,
                               uint64_t* dPrimIdRanges,
                               PrimitiveId* dPrimitiveIds,
                               uint32_t* dEndPointIndices)
    : transformIdCount(transformIdCount)
    , dTransformIds(dTransformIds)
    , dPrimIdRanges(dPrimIdRanges)
    , dPrimitiveIds(dPrimitiveIds)
    , dEndPointIndices(dEndPointIndices)
{}

__device__ __forceinline__
uint32_t EndpointFinder::FindEndPointIndex(TransformId transformId,
                                           PrimitiveId primitiveId) const
{
    // Binary Search the transform
    float index;
    bool found = GPUFunctions::BinarySearchInBetween(index, transformId, dTransformIds,
                                                     transformIdCount);
    assert(found);

    // Acquire Primitive Range for that transform
    uint32_t indexInt = static_cast<uint32_t>(index);
    uint32_t start = dPrimIdRanges[indexInt];
    uint32_t end = dPrimIdRanges[indexInt + 1];

    // Binary Search the primitiveId
    found = GPUFunctions::BinarySearchInBetween(index, primitiveId, 
                                                dPrimitiveIds + start,
                                                (end - start));
    assert(found);
    // Binary Search the primitiveId
    return dEndPointIndices[static_cast<uint32_t>(index)];
}
