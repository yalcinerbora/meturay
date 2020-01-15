#pragma once


#include "GPUAcceleratorP.cuh"
#include "DeviceMemory.h"
#include "CudaConstants.h"

#include "GPUAcceleratorBVHKC.cuh"

template<class PGroup>
class GPUAccBVHBatch;

template <class PGroup>
class GPUAccBVHGroup final
    : public GPUAcceleratorGroup<PGroup>
{
    ACCELERATOR_TYPE_NAME("BasicBVH", PGroup);

    public:
        using LeafData                      = PGroup::LeafData;

    private:
        // CPU Memory
        std::vector<PrimitiveRangeList>     primitiveRanges;
        std::vector<HitKeyList>             primitiveMaterialKeys;

        std::map<uint32_t, uint32_t>        idLookup;

        // GPU Memory
        DeviceMemory                        memory;
        const BVHNode<LeafData>**           dBVHLists;

        friend class                        GPUAccBVHBatch<PGroup>;

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
        void                            ConstructAccelerators(const CudaSystem&) override;
        void                            ConstructAccelerator(uint32_t surface,
                                                             const CudaSystem&) override;
        void                            ConstructAccelerators(const std::vector<uint32_t>& surfaces,
                                                              const CudaSystem&) override;
        void                            DestroyAccelerators(const CudaSystem&) override;
        void                            DestroyAccelerator(uint32_t surface,
                                                           const CudaSystem&) override;
        void                            DestroyAccelerators(const std::vector<uint32_t>& surfaces,
                                                            const CudaSystem&) override;

        size_t                          UsedGPUMemory() const override;
        size_t                          UsedCPUMemory() const override;
};

template<class PGroup>
class GPUAccBVHBatch final
    : public GPUAcceleratorBatch<GPUAccBVHGroup<PGroup>, PGroup>
{
    public:
        // Constructors & Destructor
                            GPUAccBVHBatch(const GPUAcceleratorGroupI&,
                                              const GPUPrimitiveGroupI&);
                            ~GPUAccBVHBatch() = default;

        // Interface
        // Type(as string) of the accelerator group
        const char*         Type() const override;
        // Kernel Logic
        void                Hit(const CudaGPU&,
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

#include "GPUAcceleratorBVH.hpp"