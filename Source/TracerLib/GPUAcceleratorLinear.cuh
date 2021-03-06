#pragma once
/**

Linear Accelerator Implementation

This is actually not an accelerator
it traverses the "constructed" (portionized)
group of primitives and calls intersection functions
one by one

It is here for sinple scenes and objects in which
tree constructio would provide additional overhead.

*/

#include <array>

#include "RayLib/SceneStructs.h"
#include "RayLib/Vector.h"
#include "RayLib/Constants.h"

#include "DeviceMemory.h"
#include "GPUAcceleratorP.cuh"
#include "GPUPrimitiveI.h"
#include "CudaConstants.h"
#include "TypeTraits.h"

#include "GPUAcceleratorLinearKC.cuh"

// This should be an array?
// Most of the time each accelerator will be constructred with a
// Singular primitive batch, it should be better to put size constraint
//using SurfaceDataList = std::vector<uint32_t>;
using SurfaceMaterialPairs = std::array<Vector2ul, SceneConstants::MaxPrimitivePerSurface>;

template<class PGroup>
class GPUAccLinearBatch;

template <class PGroup>
class GPUAccLinearGroup final
    : public GPUAcceleratorGroup<PGroup>
{
    ACCELERATOR_TYPE_NAME("Linear", PGroup);

    public:
        using LeafData                      = PGroup::LeafData;

    private:
        // CPU Memory
        std::vector<PrimitiveRangeList>     primitiveRanges;
        std::vector<HitKeyList>             primitiveMaterialKeys;
        std::vector<Vector2ul>              accRanges;
        std::map<uint32_t, uint32_t>        idLookup;

        // GPU Memory
        DeviceMemory                        memory;
        Vector2ul*                          dAccRanges;
        LeafData*                           dLeafList;

        friend class                        GPUAccLinearBatch<PGroup>;

    protected:

    public:
        // Constructors & Destructor
                                        GPUAccLinearGroup(const GPUPrimitiveGroupI&,
                                                          const TransformStruct*);
                                        ~GPUAccLinearGroup() = default;

        // Interface
        // Type(as string) of the accelerator group
        const char*                     Type() const override;
        // Loads required data to CPU cache for
        SceneError                      InitializeGroup(// Accelerator Option Node
                                                        const SceneNodePtr& node,
                                                        // Map of hit keys for all materials
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
class GPUAccLinearBatch final
    : public GPUAcceleratorBatch<GPUAccLinearGroup<PGroup>, PGroup>
{
    public:
        // Constructors & Destructor
                            GPUAccLinearBatch(const GPUAcceleratorGroupI&,
                                              const GPUPrimitiveGroupI&);
                            ~GPUAccLinearBatch() = default;

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

class GPUBaseAcceleratorLinear final : public GPUBaseAcceleratorI
{
    public:
        static const char*              TypeName();

    private:
        DeviceMemory                    leafMemory;
        DeviceMemory                    rayLocMemory;

        // GPU
        const BaseLeaf*                 dLeafs;
        uint32_t*                       dPrevLocList;

        // CPU
        std::map<uint32_t, uint32_t>    innerIds;
        uint32_t                        leafCount;

    protected:
    public:
        // Constructors & Destructor
                                        GPUBaseAcceleratorLinear();
                                        GPUBaseAcceleratorLinear(const GPUBaseAcceleratorLinear&) = delete;
        GPUBaseAcceleratorLinear&       operator=(const GPUBaseAcceleratorLinear&) = delete;
                                        ~GPUBaseAcceleratorLinear() = default;

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


        SceneError                  Initialize(// Accelerator Option Node
                                               const SceneNodePtr& node,
                                               // List of surface to transform id hit key mappings
                                               const std::map<uint32_t, BaseLeaf>&) override;
        SceneError                  Change(// List of only changed surface to transform id hit key mappings
                                           const std::map<uint32_t, BaseLeaf>&) override;

        TracerError                 Constrcut(const CudaSystem&) override;
        TracerError                 Destruct(const CudaSystem&) override;
};

#include "GPUAcceleratorLinear.hpp"