#pragma once
/**

*/

#include <cstdint>

#include "RayLib/HitStructs.h"
#include "RayLib/Ray.h"

#include "NodeListing.h"

struct RayGMem;
struct SceneError;
struct TextureStruct;

class CudaGPU;
class SceneNodeI;
class RNGMemory;
class ImageMemory;
class GPUPrimitiveGroupI;
class GPUTransformI;
class GPUMediumI;
class RNGeneratorGPUI;

struct UVSurface;

using TextureNodeMap = std::map<uint32_t, TextureStruct>;

// Defines dynamic inheritance syle of interface for each material
// on the group, normally static inheritance is used to evaluate materials
// However user can use this device classes on non-templated kernels
// for material evaluation
// By default these are NOT allocated a tracer should specifically ask
// material group to generate those
//
// Also these functions are only supports a single surface structure
// "GPU meta surface" which has normal, position and UV
class GPUMaterialI
{
    public:
    virtual                         ~GPUMaterialI() = default;
    // Interface
    __device__ virtual bool         IsEmissive() const = 0;
    __device__ virtual float        Specularity(const UVSurface& surface) const = 0;
    __device__ virtual Vector3f     Sample(// Sampled Output
                                           RayF& wo,                       // Out direction
                                           float& pdf,                     // PDF for Monte Carlo
                                           const GPUMediumI*& outMedium,
                                           // Input
                                           const Vector3& wi,              // Incoming Radiance
                                           const Vector3& pos,             // Position
                                           const GPUMediumI& m,
                                           //
                                           const UVSurface& surface,  // Surface info (normals uvs etc.)
                                           // I-O
                                           RNGeneratorGPUI& rng) const = 0;
    __device__ virtual Vector3f     Emit(// Input
                                         const Vector3& wo,      // Outgoing Radiance
                                         const Vector3& pos,     // Position
                                         const GPUMediumI& m,
                                         //
                                         const UVSurface& surface) const = 0;
    __device__ virtual Vector3f     Evaluate(// Input
                                             const Vector3& wo,              // Outgoing Radiance
                                             const Vector3& wi,              // Incoming Radiance
                                             const Vector3& pos,             // Position
                                             const GPUMediumI& m,
                                             //
                                             const UVSurface& surface) const = 0;

    __device__ virtual float       Pdf(// Input
                                       const Vector3& wo,      // Outgoing Radiance
                                       const Vector3& wi,
                                       const Vector3& pos,     // Position
                                       const GPUMediumI& m,
                                       //
                                       const UVSurface& surface) const = 0;
};

// Defines the same type materials
// Logics consists of loading certain material
// This struct holds the material data in a batched fashion (textures arrays etc)
// These are singular and can be shared by multiple accelerator batches
class GPUMaterialGroupI
{
    public:
        virtual                             ~GPUMaterialGroupI() = default;

        // Interface
        // Type (as string) of the primitive group
        virtual const char*                 Type() const = 0;
        // Allocates and Generates data
        virtual SceneError                  InitializeGroup(const NodeListing& materialNodes,
                                                            const TextureNodeMap& textures,
                                                            const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                            double time, const std::string& scenePath) = 0;
        // Changes the Generated data
        virtual SceneError                  ChangeTime(const NodeListing& materialNodes, double time,
                                                       const std::string& scenePath) = 0;
        virtual TracerError                 ConstructTextureReferences() = 0;

        // Material Queries
        virtual bool                        HasMaterial(uint32_t materialId) const = 0;
        virtual uint32_t                    InnerId(uint32_t materialId) const = 0;
        virtual const CudaGPU&              GPU() const = 0;

        // Total used GPU memory, this includes static textures
        virtual size_t                      UsedGPUMemory() const = 0;
        virtual size_t                      UsedCPUMemory() const = 0;
        virtual size_t                      UsedGPUMemory(uint32_t materialId) const = 0;
        virtual size_t                      UsedCPUMemory(uint32_t materialId) const = 0;
        // NEE Related
        virtual bool                        CanBeSampled() const { return true; }
        // Post initialization
        virtual void                        AttachGlobalMediumArray(const GPUMediumI* const*,
                                                                    uint32_t baseMediumIndex) = 0;

        // Returns how many different sampling strategy this material has
        virtual uint8_t                     SampleStrategyCount() const = 0;
        // Returns the cached textures
        virtual uint8_t                     UsedTextureCount() const = 0;
        virtual std::vector<uint32_t>       UsedTextureIds() const = 0;

        // Dynamic Inheritance Generation
        virtual void                        GeneratePerMaterialInterfaces() = 0;
        virtual const GPUMaterialI**        GPUMaterialInterfaces() const = 0;
        virtual bool                        CanSupportDynamicInheritance() const = 0;
};
