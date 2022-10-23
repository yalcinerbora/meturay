#pragma once

#include <set>
#include "RayLib/ArrayPortion.h"
#include "RayLib/HitStructs.h"
#include "RayLib/TracerStructs.h"
#include "RayLib/Vector.h"
#include "RayLib/BitManipulation.h"

#include "GPUWorkI.h"

class CudaSystem;
class RNGeneratorCPUI;
class GPUTransformI;
struct RayGMem;

template<class T>
using RayPartitions = std::set<ArrayPortion<T>>;

template<class T>
using RayPartitionsMulti = std::set<MultiArrayPortion<T>>;

class RayCasterI
{
    public:
        virtual                             ~RayCasterI() = default;

        // Static Helpers
        static Vector2i                     DetermineMaxBitFromId(const Vector2i& maxIds);
        static RayPartitionsMulti<uint32_t> PartitionOutputRays(uint32_t& totalOutRay,
                                                                const RayPartitions<uint32_t>& inPartitions,
                                                                const WorkBatchMap& workMap);
        // Interface
        virtual TracerError             ConstructAccelerators(const GPUTransformI** dTransforms,
                                                              uint32_t identityTransformIndex) = 0;

        virtual void                    HitRays() = 0;
        // Work Partition and Call Functions
        virtual RayPartitions<uint32_t> PartitionRaysWRTWork(bool skipInternalPartitioning = false) = 0;
        virtual void                    WorkRays(const WorkBatchMap& workMap,
                                                 const RayPartitionsMulti<uint32_t>& outPortions,
                                                 const RayPartitions<uint32_t>& inPartitions,
                                                 RNGeneratorCPUI& rngCPU,
                                                 uint32_t totalRayOut,
                                                 HitKey baseBoundMatKey) = 0;
        // Ray Related
        virtual uint32_t                CurrentRayCount() const = 0;
        virtual void                    ResizeRayOut(uint32_t rayCount,
                                                     HitKey baseBoundMatKey) = 0;
        // RayMemory Fetch
        // Input
        virtual const RayId*            SortedRayIds() const = 0;
        // Unsorted Data
        virtual const RayGMem*          RaysIn() const = 0;
        virtual const HitKey*           WorkKeys() const = 0;
        virtual const PrimitiveId*      PrimitiveIds() const = 0;
        virtual const TransformId*      TransformIds() const = 0;
        virtual const HitStructPtr      HitSturctPtr() const = 0;
        // Output
        virtual RayGMem*                RaysOut() = 0;
        virtual HitKey*                 KeysOut() = 0;
        // Misc
        virtual void                    SwapRays() = 0;
        // Work Related
        virtual void                    OverrideWorkBits(const Vector2i newWorkBits) = 0;
        // Memory Usage
        virtual size_t                  UsedGPUMemory() const = 0;

};

inline Vector2i RayCasterI::DetermineMaxBitFromId(const Vector2i& maxIds)
{
    assert(maxIds[0] >= 0 && maxIds[1] >= 0);

    Vector2i result((maxIds[0] == 0) ? 0 : static_cast<int32_t>(Utility::FindLastSet<uint32_t>(maxIds[0]) + 1),
                    (maxIds[1] == 0) ? 0 : static_cast<int32_t>(Utility::FindLastSet<uint32_t>(maxIds[1]) + 1));
    return result;
}

inline RayPartitionsMulti<uint32_t> RayCasterI::PartitionOutputRays(uint32_t& totalOutRay,
                                                                    const RayPartitions<uint32_t>& inPartitions,
                                                                    const WorkBatchMap& workMap)
{
    RayPartitionsMulti<uint32_t> outPartitions;

    // Find total ray out
    totalOutRay = 0;
    for(auto pIt = inPartitions.crbegin();
        pIt != inPartitions.crend(); pIt++)
    {
        const auto& p = (*pIt);

        // Skip if null batch or not found material
        if(p.portionId == HitKey::NullBatch) continue;
        auto loc = workMap.find(p.portionId);
        if(loc == workMap.end()) continue;

        std::vector<size_t> offsets;
        std::vector<size_t> counts;

        // Generate Portions for each shade call
        for(const auto& wb : loc->second)
        {
            uint32_t count = (static_cast<uint32_t>(p.count) *
                              wb->OutRayCount());

            counts.push_back(count);
            offsets.push_back(totalOutRay);
            totalOutRay += count;
        }

        outPartitions.emplace(MultiArrayPortion<uint32_t>
        {
            p.portionId,
            offsets,
            counts
        });
    }
    return outPartitions;
}