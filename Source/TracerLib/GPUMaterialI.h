#pragma once
/**


*/

#include <cstdint>

#include "RayLib/HitStructs.h"
#include "RayLib/Vector.h"

#include "NodeListing.h"

struct RayGMem;
struct SceneError;

class CudaGPU;
class SceneNodeI;
class RNGMemory;
class ImageMemory;
class GPUPrimitiveGroupI;
class GPUEventEstimatorI;

// Defines the same type materials
// Logics consists of loading unloading certain material
// This struct holds the material data in a batched fashion (textures arrays etc)
// These are singular and can be shared by multiple accelrator batches
class GPUMaterialGroupI
{
    public:
        virtual                             ~GPUMaterialGroupI() = default;

        // Interface
        // Type (as string) of the primitive group
        virtual const char*                 Type() const = 0;
        // Allocates and Generates Data
        virtual SceneError                  InitializeGroup(const NodeListing& materialNodes, double time,
                                                            const std::string& scenePath) = 0;
        virtual SceneError                  ChangeTime(const NodeListing& materialNodes, double time,
                                                       const std::string& scenePath) = 0;

        // Material Queries
        virtual int                         InnerId(uint32_t materialId) const = 0;
        virtual bool                        HasCachedTextures(uint32_t materialId) const = 0;
        virtual const CudaGPU&              GPU() const = 0;

        virtual const GPUEventEstimatorI&   EventEstimator() const = 0;

        virtual size_t                      UsedGPUMemory() const = 0;
        virtual size_t                      UsedCPUMemory() const = 0;
        virtual size_t                      UsedGPUMemory(uint32_t materialId) const = 0;
        virtual size_t                      UsedCPUMemory(uint32_t materialId) const = 0;
        

        virtual uint8_t                     OutRayCount() const = 0;                
};

// Defines call action over a certain material group
// The batch further specializes over a primitive logic
// which defines how certain primitive data could be fetched
class GPUMaterialBatchI
{
    public:
        virtual                             ~GPUMaterialBatchI() = default;

        // Interface
        // Type (as string) of the primitive group
        virtual const char*                 Type() const = 0;
        // Kernel Call
        virtual void                        ShadeRays(// Output
                                                      ImageMemory& dImage,
                                                      //
                                                      HitKey* dBoundMatOut,
                                                      RayGMem* dRayOut,
                                                      void* dRayAuxOut,
                                                      //  Input
                                                      const RayGMem* dRayIn,
                                                      const void* dRayAuxIn,
                                                      const PrimitiveId* dPrimitiveIds,
                                                      const HitStructPtr dHitStructs,
                                                      // Ids
                                                      const HitKey* dMatIds,
                                                      const RayId* dRayIds,
                                                      // 
                                                      const uint32_t rayCount,
                                                      RNGMemory& rngMem) const = 0;

        // Every MaterialBatch is available for a specific primitive / material data
        virtual const GPUPrimitiveGroupI&   PrimitiveGroup() const = 0;
        virtual const GPUMaterialGroupI&    MaterialGroup() const = 0;

        virtual uint8_t                     OutRayCount() const = 0;
};
