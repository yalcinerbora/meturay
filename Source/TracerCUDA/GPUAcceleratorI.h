#pragma once
/**

Base Interface for GPU accelerators

*/

#include <map>
#include <set>
#include <array>
#include <cstdint>
#include <functional>

#include "RayLib/HitStructs.h"
#include "RayLib/SceneStructs.h"
#include "RayLib/AABB.h"

#include "AcceleratorFunctions.h"

struct RayGMem;
struct SceneError;

class CudaGPU;
class CudaSystem;
class GPUPrimitiveGroupI;
class GPUMaterialGroupI;
class SceneNodeI;
class GPUTransformI;

using SceneNodePtr = std::unique_ptr<SceneNodeI>;
using SurfaceAABBList = std::map<uint32_t, AABB3f>;

// Accelerator Group defines same type of accelerators
// This struct holds accelerator data
// Unlike material group there is one to one relationship between
// Accelerator batch and group since accelerator is strongly tied with primitive
// However interface is split for consistency (to be like material group/batch)
class GPUAcceleratorGroupI
{
    public:
        virtual                 ~GPUAcceleratorGroupI() = default;

        // Interface
        // Type(as string) of the accelerator group
        virtual const char*     Type() const = 0;
        // Loads required data to CPU cache for
        virtual SceneError      InitializeGroup(// Accelerator Option Node
                                                const SceneNodePtr& node,
                                                // Map of hit keys for all materials
                                                // w.r.t matId and primitive type
                                                const std::map<TypeIdPair, HitKey>&,
                                                // List of surface/material
                                                // pairings that uses this accelerator type
                                                // and primitive type
                                                const std::map<uint32_t, IdPairs>& pairingList,
                                                const std::vector<uint32_t>& transformList,
                                                double time) = 0;

        // Surface Queries
        virtual uint32_t                InnerId(uint32_t surfaceId) const = 0;

        // Batched and singular construction
        virtual TracerError             ConstructAccelerators(const CudaSystem&) = 0;
        virtual TracerError             ConstructAccelerator(uint32_t surface,
                                                             const CudaSystem&) = 0;
        virtual TracerError             ConstructAccelerators(const std::vector<uint32_t>& surfaces,
                                                              const CudaSystem&) = 0;

        virtual TracerError             DestroyAccelerators(const CudaSystem&) = 0;
        virtual TracerError             DestroyAccelerator(uint32_t surface,
                                                           const CudaSystem&) = 0;
        virtual TracerError             DestroyAccelerators(const std::vector<uint32_t>& surfaces,
                                                            const CudaSystem&) = 0;

        virtual size_t                  UsedGPUMemory() const = 0;
        virtual size_t                  UsedCPUMemory() const = 0;

        // Kernel Logic
        virtual void                    Hit(const CudaGPU&,
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
                                            const uint32_t rayCount) const = 0;

        virtual const SurfaceAABBList&      AcceleratorAABBs() const = 0;
        virtual const GPUPrimitiveGroupI&   PrimitiveGroup() const = 0;
        virtual void                        AttachGlobalTransformArray(const GPUTransformI** deviceTransforms,
                                                                       uint32_t identityTransformIndex) = 0;
};

class GPUBaseAcceleratorI
{
    public:
        virtual                 ~GPUBaseAcceleratorI() = default;

        // Interface
        // Type(as string) of the accelerator group
        virtual const char*     Type() const = 0;
        // Get ready for hit loop
        virtual void            GetReady(const CudaSystem& system, 
                                         uint32_t rayCount) = 0;
        // Base accelerator only points to the next accelerator key.
        // It can return invalid key,
        // which is either means data is out of bounds or ray is invalid.
        virtual void            Hit(const CudaSystem&,
                                    // Output
                                    HitKey* dMaterialKeys,
                                    // Inputs
                                    const RayGMem* dRays,
                                    const RayId* dRayIds,
                                    const uint32_t rayCount) const = 0;

        // Initialization
        virtual SceneError      Initialize(// Accelerator Option Node
                                           const SceneNodePtr& node,
                                           // List of surface to leaf accelerator ids
                                           const std::map<uint32_t, HitKey>&) = 0;

        // Construction & Destruction
        virtual TracerError     Constrcut(const CudaSystem&,
                                          // List of surface AABBs
                                          const SurfaceAABBList&) = 0;
        virtual TracerError     Destruct(const CudaSystem&) = 0;
};