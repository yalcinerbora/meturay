#pragma once

#include <cub/cub.cuh>
#include <vector>
#include <queue>

#include "RayLib/MemoryAlignment.h"
#include "RayLib/CPUTimer.h"

#include "DeviceMemory.h"
#include "CudaConstants.h"

#include "GPUAcceleratorP.cuh"
#include "GPUAcceleratorBVHKC.cuh"

template<class LeafData>
using AllLeafDataCPU = std::vector<std::vector<BVHNode<LeafData>>>;

template<class PGroup>
class GPUAccBVHBatch;

static inline SplitAxis DetermineNextSplit(SplitAxis split, const AABB3f& aabb)
{
    SplitAxis nextSplit = static_cast<SplitAxis>((static_cast<int>(split) + 1) %
                                                 static_cast<int>(SplitAxis::END));
    int splitIndex = static_cast<int>(nextSplit);
    // Skip this split if it is very tight (compared to other axis)
    Vector3 diff = aabb.Max() - aabb.Min();
    // AABB is like a 2D AABB skip this axis
    if(std::abs(diff[splitIndex]) < 0.001f)
        nextSplit = static_cast<SplitAxis>((static_cast<int>(nextSplit) + 1) %
                                           static_cast<int>(SplitAxis::END));
    return nextSplit;
}

template <class PGroup>
class GPUAccBVHGroup final
    : public GPUAcceleratorGroup<PGroup>
{
    ACCELERATOR_TYPE_NAME("BasicBVH", PGroup);

    public:
        using LeafData                      = PGroup::LeafData;

    private:
        static constexpr const uint32_t     Threshold_CPU_GPU = 32'768;
        static constexpr size_t             AlignByteCount = 16;

        // CPU Memory
        std::vector<PrimitiveRangeList>     primitiveRanges;
        std::vector<HitKeyList>             primitiveMaterialKeys;
        std::vector<uint8_t>                bvhDepths;

        std::map<uint32_t, uint32_t>        idLookup;
        // GPU Memory
        DeviceMemory                        bvhListMemory;
        std::vector<DeviceMemory>           bvhMemories;
        const BVHNode<LeafData>**           dBVHLists;

        friend class                        GPUAccBVHBatch<PGroup>;

        // Recursive Construction
        HitKey                              FindHitKey(uint32_t accIndex, PrimitiveId id);
        void                                GenerateBVHNode(// Output
                                                            size_t& splitLoc,
                                                            BVHNode<LeafData>& node,
                                                            //Temp Memory
                                                            void* dTemp,
                                                            size_t tempMemSize,
                                                            uint32_t* dPartitionSplitOut,
                                                            uint32_t* dIndicesTemp,
                                                            // Index Data                                             
                                                            uint32_t* dIndicesIn,
                                                            // Constants
                                                            const uint64_t* dPrimIds,
                                                            const Vector3f* dPrimCenters,
                                                            const AABB3f* dAABBs,
                                                            uint32_t accIndex,
                                                            const CudaGPU& gpu,
                                                            // Call Related Args
                                                            uint32_t parentIndex,
                                                            SplitAxis axis,
                                                            size_t start, size_t end);

    public:
        // Constructors & Destructor
                                        GPUAccBVHGroup(const GPUPrimitiveGroupI&,
                                                       const TransformStruct*);
                                        ~GPUAccBVHGroup() = default;

        // Interface
        // Type(as string) of the accelerator group
        const char*                     Type() const override;
        // Loads required data to CPU cache for
        SceneError                      InitializeGroup(// Map of hit keys for all materials
                                                        // w.r.t matId and primitive type
                                                        const std::map<TypeIdPair, HitKey>&,
                                                        // List of surface/material
                                                        // pairings that uses this accelerator type
                                                        // and primitive type
                                                        const std::map<uint32_t, IdPairs>& parList,
                                                        double time) override;
        SceneError                      ChangeTime(// Map of hit keys for all materials
                                                   // w.r.t matId and primitive type
                                                   const std::map<TypeIdPair, HitKey>&,
                                                   // List of surface/material
                                                   // pairings that uses this accelerator type
                                                   // and primitive type
                                                   const std::map<uint32_t, IdPairs>& parList,
                                                   double time) override;

        // Surface Queries
        uint32_t                        InnerId(uint32_t surfaceId) const override;

        // Batched and singular construction
        TracerError                     ConstructAccelerators(const CudaSystem&) override;
        TracerError                     ConstructAccelerator(uint32_t surface,
                                                             const CudaSystem&) override;
        TracerError                     ConstructAccelerators(const std::vector<uint32_t>& surfaces,
                                                              const CudaSystem&) override;
        TracerError                     DestroyAccelerators(const CudaSystem&) override;
        TracerError                     DestroyAccelerator(uint32_t surface,
                                                           const CudaSystem&) override;
        TracerError                     DestroyAccelerators(const std::vector<uint32_t>& surfaces,
                                                            const CudaSystem&) override;

        size_t                          UsedGPUMemory() const override;
        size_t                          UsedCPUMemory() const override;
};

template<class PGroup>
class GPUAccBVHBatch final
    : public GPUAcceleratorBatch<GPUAccBVHGroup<PGroup>, PGroup>
{
    public:
        static constexpr bool   USE_STACK = true;

        // Constructors & Destructor
                                GPUAccBVHBatch(const GPUAcceleratorGroupI&,
                                                  const GPUPrimitiveGroupI&);
                                ~GPUAccBVHBatch() = default;

        // Interface
        // Type(as string) of the accelerator group
        const char*             Type() const override;
        // Kernel Logic
        void                    Hit(const CudaGPU&,
                                    // O
                                    HitKey* dMaterialKeys,
                                    PrimitiveId* dPrimitiveIds,
                                    HitStructPtr dHitStructs,
                                    // I-O
                                    RayGMem* dRays,
                                    // Input
                                    const TransformId* dTransformIds,
                                    const RayId* dRayIds,
                                    const HitKey* dAcceleratorKeys,
                                    const uint32_t rayCount) const override;
};


class GPUBaseAcceleratorBVH final : public GPUBaseAcceleratorI
{
    public:
        static const char*              TypeName();

    private:
        static constexpr size_t         AlignByteCount = 16;
        // GPU Memory
        DeviceMemory                    bvhMemory;
        DeviceMemory                    rayStateMemory;
        // GPU Ptrs
        const BVHNode<BaseLeaf>*        dBVH;
        uint32_t*                       dRayStates;
        uint32_t*                       dPrevBVHIndex;

        // CPU Memory
        std::map<uint32_t, uint32_t>    idLookup;
        std::vector<BaseLeaf>           leafs;

        static void                     GenerateBaseBVHNode(// Output
                                                            size_t& splitLoc,
                                                            BVHNode<BaseLeaf>& node,
                                                            // Index Data                                             
                                                            uint32_t* surfaceIndices,
                                                            // Constants
                                                            const BaseLeaf* leafs,
                                                            const Vector3f* centers,
                                                            // Call Related Args
                                                            uint32_t parentIndex,
                                                            SplitAxis axis,
                                                            size_t start, size_t end);

    public:
        // Constructors & Destructor
                                        GPUBaseAcceleratorBVH();
                                        GPUBaseAcceleratorBVH(const GPUBaseAcceleratorBVH&) = delete;
        GPUBaseAcceleratorBVH&          operator=(const GPUBaseAcceleratorBVH&) = delete;
                                        ~GPUBaseAcceleratorBVH() = default;

        // Interface
        // Type(as string) of the accelerator group
        const char*                 Type() const override;

        // Get ready for hit loop
        void                        GetReady(const CudaSystem& system, 
                                             uint32_t rayCount) override;
        // Base accelerator only points to the next accelerator key.
        // It can return invalid key,
        // which is either means data is out of bounds or ray is invalid.
        void                        Hit(const CudaSystem&,
                                        // Output
                                        TransformId* dTransformIds,
                                        HitKey* dAcceleratorKeys,
                                        // Inputs
                                        const RayGMem* dRays,
                                        const RayId* dRayIds,
                                        const uint32_t rayCount) const override;


        SceneError                  Initialize(// List of surface to transform id hit key mappings
                                               const std::map<uint32_t, BaseLeaf>&) override;
        SceneError                  Change(// List of only changed surface to transform id hit key mappings
                                           const std::map<uint32_t, BaseLeaf>&) override;

        TracerError                 Constrcut(const CudaSystem&) override;
        TracerError                 Destruct(const CudaSystem&) override;
};

#include "GPUAcceleratorBVH.hpp"