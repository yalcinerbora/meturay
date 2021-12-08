#pragma once
/**

General Device memory manager for ray and it's auxiliary data

*/

#include <set>

#include "RayLib/ArrayPortion.h"
#include "RayLib/HitStructs.h"

#include "DeviceMemory.h"
#include "RayStructs.h"

class CudaGPU;

template<class T>
using RayPartitions = std::set<ArrayPortion<T>>;

template<class T>
using RayPartitionsMulti = std::set<MultiArrayPortion<T>>;

class RayMemory
{
    private:
        // Leader GPU device that is responsible for
        // partitioning and sorting the ray data
        // (Only useful in multi-GPU systems)
        const CudaGPU&              leaderDevice;

        // Ray Related
        DeviceMemory                memIn;
        DeviceMemory                memOut;
        // In rays are going to enter material kernels
        RayGMem*                    dRayIn;
        // Those kernels will output one or multiple rays
        // Each material has a predefined max ray output
        // Out is allocated accordingly then materials fill it
        RayGMem*                    dRayOut;
        //---------------------------------------------------------
        // Hit Related
        // Entire Hit related memory is allocated in bulk.
        DeviceMemory                memHit;
        // MatKey holds the work batch id and material group local id
        // This is used to sort rays to match material kernels
        HitKey*                     dWorkKeys;
        // Transform of the hit
        // Base accelerator fill this value with a potential hit id
        // Leaf accelerators will transform rays to find hit
        TransformId*                dTransformIds;
        // Primitive Id of the hit
        // Inner accelerators fill this value with a primitive group local id
        // Primitive group id can be determined by work group
        PrimitiveId*                dPrimitiveIds;
        // Custom hit Structure allocation pointer
        // This pointer is capable of holding data for all
        // hit structures currently active
        // (i.e. it holds Vec2 bary coords for triangle primitives,
        // hold position for spheres (maybe spherical coords in order to save space).
        // or other custom value for a custom primitive (spline params maybe i dunno)
        HitStructPtr                dHitStructs;
        // --
        // Double buffer and temporary memory for sorting
        // Key/Index pair (key can either be accelerator or material)
        void*                       dTempMemory;
        RayId*                      dIds0, *dIds1;
        HitKey*                     dKeys0, *dKeys1;
        // Current pointers to the double buffer
        // In hit portion of the code it holds accelerator ids etc.
        HitKey*                     dCurrentKeys;
        RayId*                      dCurrentIds;

        // Cub Requires actual tempMemory size
        // It sometimes crash if you give larger memory size
        // I dunno why
        size_t                      cubIfMemSize;
        size_t                      cubSortMemSize;

    public:
        // Constructors & Destructor
                                RayMemory(const CudaGPU& leaderDevice);
                                RayMemory(const RayMemory&) = delete;
                                RayMemory(RayMemory&&) = default;
        RayMemory&              operator=(const RayMemory&) = delete;
        RayMemory&              operator=(RayMemory&&) = delete;
                                ~RayMemory() = default;

        // Accessors
        // Ray In
        RayGMem*                    Rays();
        // Ray Out
        RayGMem*                    RaysOut();

        // Hit Related
        HitStructPtr                HitStructs();
        const HitStructPtr          HitStructs() const;
        HitKey*                     WorkKeys();
        const HitKey*               WorkKeys() const;
        TransformId*                TransformIds();
        const TransformId*          TransformIds() const;
        PrimitiveId*                PrimitiveIds();
        const PrimitiveId*          PrimitiveIds() const;
        // For sorting
        HitKey*                     CurrentKeys();
        const HitKey*               CurrentKeys() const;
        RayId*                      CurrentIds();
        const RayId*                CurrentIds() const;

        // Misc
        const CudaGPU&              LeaderDevice() const;

        // Memory Allocation
        void                        ResetHitMemory(TransformId identityTransformIndex,
                                                   uint32_t rayCount, size_t hitStructSize);
        void                        ResizeRayOut(uint32_t rayCount, HitKey baseBoundMatKey);
        void                        SwapRays();

        // Utilities
        size_t                      TotalMemorySize();

        // Common Functions
        // Sorts the hit list for multi-kernel calls
        void                        SortKeys(RayId*& ids, HitKey*& keys,
                                             uint32_t count,
                                             const Vector2i& bitRange);
        // Partitions the segments for multi-kernel calls
        RayPartitions<uint32_t>     Partition(uint32_t rayCount);
        // Initialize HitIds and Indices
        void                        FillMatIdsForSort(uint32_t rayCount);
        // Mem Usage
        size_t                      UsedGPUMemory() const;
};

inline const CudaGPU& RayMemory::LeaderDevice() const
{
    return leaderDevice;
}

inline RayGMem* RayMemory::Rays()
{
    return dRayIn;
}

inline RayGMem* RayMemory::RaysOut()
{
    return dRayOut;
}

inline HitStructPtr RayMemory::HitStructs()
{
    return dHitStructs;
}

inline const HitStructPtr RayMemory::HitStructs() const
{
    return dHitStructs;
}

inline HitKey* RayMemory::WorkKeys()
{
    return dWorkKeys;
}

inline const HitKey* RayMemory::WorkKeys() const
{
    return dWorkKeys;
}

inline TransformId* RayMemory::TransformIds()
{
    return dTransformIds;
}

inline const TransformId* RayMemory::TransformIds() const
{
    return dTransformIds;
}

inline PrimitiveId* RayMemory::PrimitiveIds()
{
    return dPrimitiveIds;
}

inline const PrimitiveId* RayMemory::PrimitiveIds() const
{
    return dPrimitiveIds;
}
//
inline HitKey* RayMemory::CurrentKeys()
{
    return dCurrentKeys;
}
inline const HitKey* RayMemory::CurrentKeys() const
{
    return dCurrentKeys;
}
inline RayId* RayMemory::CurrentIds()
{
    return dCurrentIds;
}

inline const RayId* RayMemory::CurrentIds() const
{
    return dCurrentIds;
}

inline size_t RayMemory::UsedGPUMemory() const
{
    return (memIn.Size() +
            memOut.Size() +
            memHit.Size());
}