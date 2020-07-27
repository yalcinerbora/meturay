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
class GPUTextureCacheI;

// METURay only supports 64 texture per material
using TextureMask = uint64_t;

// Defines the same type materials
// Logics consists of loading certain material
// This struct holds the material data in a batched fashion (textures arrays etc)
// These are singular and can be shared by multiple accelrator batches
class GPUMaterialGroupI
{
    public:
        virtual                             ~GPUMaterialGroupI() = default;

        // Interface
        // Type (as string) of the primitive group
        virtual const char*                 Type() const = 0;
        // Allocates and Generates data
        virtual SceneError                  InitializeGroup(const NodeListing& materialNodes, double time,
                                                            const std::string& scenePath) = 0;
        // Changes the Generated data
        virtual SceneError                  ChangeTime(const NodeListing& materialNodes, double time,
                                                       const std::string& scenePath) = 0;

        // Material Queries
        virtual bool                        HasMaterial(uint32_t materialId) const = 0;
        virtual uint32_t                    InnerId(uint32_t materialId) const = 0;
        virtual bool                        HasCachedTextures(uint32_t materialId) const = 0;
        virtual const CudaGPU&              GPU() const = 0;

        // Total used GPU memory, this includes static textures
        virtual size_t                      UsedGPUMemory() const = 0;
        virtual size_t                      UsedCPUMemory() const = 0;
        virtual size_t                      UsedGPUMemory(uint32_t materialId) const = 0;
        virtual size_t                      UsedCPUMemory(uint32_t materialId) const = 0;
        
        // NEE Related
        virtual bool                        IsLightGroup() const = 0;
        virtual bool                        IsEmissiveGroup() const = 0;

        // Returns how many different sampling strategy this material has
        virtual uint8_t                     SampleStrategyCount() const = 0;
        // Returns the cached textures
        virtual uint8_t                     UsedTextureCount() const = 0;
        virtual std::vector<uint32_t>       UsedTextureIds() const = 0;
        virtual TextureMask                 CachedTextures() const = 0;
};

// Additional Interface for light materials
class LightMaterialI
{
    public:
        virtual                             ~LightMaterialI() = default;
        // Interface
        virtual const GPUDistribution2D&    LuminanceDistribution(uint32_t materialId) const = 0;
};

