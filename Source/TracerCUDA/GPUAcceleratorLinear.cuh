#pragma once
/**

Linear Accelerator Implementation

This is actually not an accelerator
it traverses the "constructed"
group of primitives and calls intersection functions
one by one

It is here for simple scenes and objects in which
tree construction would provide additional overhead.

*/

#include <array>

#include "RayLib/SceneStructs.h"
#include "RayLib/Vector.h"
#include "RayLib/Constants.h"
#include "RayLib/MemoryAlignment.h"

#include "DeviceMemory.h"
#include "CudaSystem.h"
#include "TypeTraits.h"

#include "GPUPrimitiveP.cuh"
#include "GPUAcceleratorP.cuh"
#include "GPUPrimitiveI.h"
#include "RNGSobol.cuh"
#include "ParallelSequence.cuh"

#include "GPUAcceleratorLinearKC.cuh"

// This should be an array?
// Most of the time each accelerator will be constructed with a
// Singular primitive batch, it should be better to put size constraint
//using SurfaceDataList = std::vector<uint32_t>;
using SurfaceMaterialPairs = std::array<Vector2ul, SceneConstants::MaxPrimitivePerSurface>;

template <class PGroup>
class GPUAccLinearGroup final
    : public GPUAcceleratorGroup<PGroup>
{
    ACCELERATOR_TYPE_NAME("Linear", PGroup);

    public:
        using LeafData                      = typename PGroup::LeafData;

    private:
        // CPU Memory
        std::vector<PrimitiveIdList>        primitiveIds;
        std::vector<PrimitiveRangeList>     primitiveRanges;
        std::vector<HitKeyList>             primitiveMaterialKeys;
        std::vector<Vector2ul>              accRanges;
        std::vector<bool>                   keyExpandOption;

        std::map<uint32_t, uint32_t>        idLookup;
        SurfaceAABBList                     surfaceAABBs;

        // GPU Memory
        DeviceMemory                        memory;
        Vector2ul*                          dAccRanges;
        LeafData*                           dLeafList;
        TransformId*                        dAccTransformIds;

    protected:

    public:
        // Constructors & Destructor
                            GPUAccLinearGroup(const GPUPrimitiveGroupI&);
                            ~GPUAccLinearGroup() = default;

        // Interface
        // Type(as string) of the accelerator group
        const char*         Type() const override;
        // Loads required data to CPU cache for
        SceneError          InitializeGroup(// Accelerator Option Node
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

        // Logic
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
        size_t                  AcceleratorCount() const override;
        // Arbitrary Position Fetching
        size_t                  TotalPrimitiveCount() const override;
        float                   TotalApproximateArea(const CudaSystem&) const override;
        void                    SampleAreaWeightedPoints(// Outs
                                                         Vector3f* dPositions,
                                                         Vector3f* dNormals,
                                                         // I-O
                                                         RNGSobolCPU& rngCPU,
                                                         // Inputs
                                                         uint32_t surfacePatchCount,
                                                         const CudaSystem&) const override;
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
        std::map<uint32_t, uint32_t>    idLookup;
        uint32_t                        leafCount;
        AABB3f                          sceneAABB;

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
                                        HitKey* dAcceleratorKeys,
                                        // Inputs
                                        const RayGMem* dRays,
                                        const RayId* dRayIds,
                                        const uint32_t rayCount) const override;

        SceneError                  Initialize(// Accelerator Option Node
                                               const SceneNodePtr& node,
                                               // List of surface to leaf accelerator ids
                                               const std::map<uint32_t, HitKey>&) override;

        TracerError                 Construct(const CudaSystem&,
                                              // List of surface AABBs
                                              const SurfaceAABBList&) override;
        TracerError                 Destruct(const CudaSystem&) override;

        const AABB3f&               SceneExtents() const override;

        size_t                      UsedGPUMemory() const override;
        size_t                      UsedCPUMemory() const override;
};

#include "GPUAcceleratorLinear.hpp"