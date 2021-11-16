#pragma once

#include "RayLib/MemoryAlignment.h"
#include "RayLib/CPUTimer.h"
#include "RayLib/Log.h"

#include "DeviceMemory.h"
#include "CudaSystem.h"

#include "GPUPrimitiveP.cuh"
#include "GPUAcceleratorP.cuh"

//#include "GPUAcceleratorOptixKC.cuh"
#include "GPUAcceleratorCommonKC.cuh"

#include "OptixSystem.h"
#include "OptixCheck.h"

class GPUAccGroupOptiXI
{
    public:
        virtual                 ~GPUAccGroupOptiXI() = default;

        // Return the hit record list for each accelerator
        virtual DeviceMemory    GetHitRecords() const = 0;
        //
        virtual void            SetOptiXSystem(const OptiXSystem*) = 0;
};

template <class PGroup>
class GPUAccOptiXGroup final
    : public GPUAcceleratorGroup<PGroup>
    , public GPUAccGroupOptiXI
{
    ACCELERATOR_TYPE_NAME("OptiX", PGroup);

    public:
        using LeafData = typename PGroup::LeafData;
        using PrimData = typename PGroup::PrimitiveData;

        struct DeviceTraversables
        {
            std::vector<DeviceLocalMemory>      tMemories;
            std::vector<OptixTraversableHandle> traversables;
        };

    private:
        // Unsued
        SurfaceAABBList                     emptyAABBList;
        // CPU Memory
        const OptiXSystem*                  optixSystem;
        //
        std::vector<PrimitiveRangeList>     primitiveRanges;
        std::vector<HitKeyList>             primitiveMaterialKeys;
        std::map<uint32_t, uint32_t>        idLookup;
        std::vector<bool>                   keyExpandOption;
        // GPU Memory
        std::vector<DeviceTraversables>     optixDataPerGPU;
        DeviceMemory                        transformIdMemory;
        TransformId*                        dAccTransformIds;
        //
        DeviceMemory                        leafAndPrimDataMemory;
        LeafData*                           dLeafs;
        PrimData*                           dPrimData;

    public:
        // Constructors & Destructor
                                GPUAccOptiXGroup(const GPUPrimitiveGroupI&);
                                GPUAccOptiXGroup(const GPUAccOptiXGroup&) = delete;
        GPUAccOptiXGroup&       operator=(const GPUAccOptiXGroup&) = delete;
                                ~GPUAccOptiXGroup() = default;

        // Interface
        // Type(as string) of the accelerator group
        const char*             Type() const override;
        // Loads required data to CPU cache for
        SceneError              InitializeGroup(// Accelerator Option Node
                                                const SceneNodePtr& node,
                                                // List of surface/material
                                                // pairings that uses this accelerator type
                                                // and primitive type
                                                const std::map<uint32_t, SurfaceDefinition>& surfaceList,
                                                double time) override;

        // Surface Queries
        uint32_t                InnerId(uint32_t surfaceId) const override;

        // Batched and singular construction
        TracerError             ConstructAccelerators(const CudaSystem&) override;
        TracerError             ConstructAccelerator(uint32_t surface,
                                                     const CudaSystem&) override;
        TracerError             ConstructAccelerators(const std::vector<uint32_t>& surfaces,
                                                      const CudaSystem&) override;
        TracerError             DestroyAccelerators(const CudaSystem&) override;
        TracerError             DestroyAccelerator(uint32_t surface,
                                                   const CudaSystem&) override;
        TracerError             DestroyAccelerators(const std::vector<uint32_t>& surfaces,
                                                    const CudaSystem&) override;

        size_t                  UsedGPUMemory() const override;
        size_t                  UsedCPUMemory() const override;

        // Kernel Logic
        void                    Hit(const CudaGPU&,
                                    // O
                                    HitKey* dMaterialKeys,
                                    TransformId* dTransformIds,
                                    PrimitiveId* dPrimitiveIds,
                                    HitStructPtr dHitStructs,
                                    // I-O
                                    RayGMem* dRays,
                                    // Input
                                    const RayId* dRayIds,
                                    const HitKey* dAcceleratorKeys,
                                    const uint32_t rayCount) const override;

        const SurfaceAABBList&  AcceleratorAABBs() const override;

        // OptiX Implementation
        // Return the hit record list for each accelerator
        DeviceMemory    GetHitRecords() const override;
        //
        void            SetOptiXSystem(const OptiXSystem*) override;

};

class GPUBaseAcceleratorOptiX final : public GPUBaseAcceleratorI
{
    public:
        static const char*              TypeName();

    private:
        static constexpr size_t         AlignByteCount = 16;

        // CPU Memory
        const OptiXSystem*              optixSystem;
        std::map<uint32_t, uint32_t>    idLookup;
        std::vector<BaseLeaf>           leafs;
        AABB3f                          sceneAABB;
        OptixTraversableHandle          baseTraversable;
        // GPU Memory
        DeviceMemory                    baseTraversableMemory;

    public:
        // Constructors & Destructor
                                        GPUBaseAcceleratorOptiX();
                                        GPUBaseAcceleratorOptiX(const GPUBaseAcceleratorOptiX&) = delete;
        GPUBaseAcceleratorOptiX&        operator=(const GPUBaseAcceleratorOptiX&) = delete;
                                        ~GPUBaseAcceleratorOptiX() = default;

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
                                        HitKey* dAcceleratorKeys,
                                        // Inputs
                                        const RayGMem* dRays,
                                        const RayId* dRayIds,
                                        const uint32_t rayCount) const override;

        SceneError                  Initialize(// Accelerator Option Node
                                               const SceneNodePtr& node,
                                               // List of surface to leaf accelerator ids
                                               const std::map<uint32_t, HitKey>&) override;

        TracerError                 Constrcut(const CudaSystem&,
                                              // List of surface AABBs
                                              const SurfaceAABBList&) override;
        TracerError                 Destruct(const CudaSystem&) override;

        const AABB3f&               SceneExtents() const override;

        // OptiX Related
        void                        SetOptiXSystem(const OptiXSystem*);
        OptixTraversableHandle      GetBaseTraversable() const;
};

#include "GPUAcceleratorOptiX.hpp"