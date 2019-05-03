#pragma once
/**

Tracer Logic:

Responsible for containing logic CUDA Tracer

This wll be wrapped by a template class which partially implements
some portions of the main code

That interface is responsible for fetching


*/

#include <cstdint>
#include <map>

#include "RayLib/Constants.h"
#include "RayLib/Vector.h"
#include "RayLib/Camera.h"
#include "RayLib/TracerStructs.h"

// Common Memory
class RayMemory;
class RNGMemory;

struct TracerError;

class GPUScene;
class GPUBaseAcceleratorI;
class GPUAcceleratorBatchI;
class GPUMaterialBatchI;

class TracerBaseLogicI
{
    public:
        virtual                 ~TracerBaseLogicI() = default;

        // Interface
        virtual TracerError     Initialize() = 0;

        // Generate Rays
        virtual uint32_t        GenerateRays(RayMemory&, RNGMemory&,
                                             const GPUScene& scene,
                                             int cameraId,
                                             int samplePerLocation,
                                             Vector2i resolution,
                                             Vector2i pixelStart = Zero2i,
                                             Vector2i pixelEnd = BaseConstants::IMAGE_MAX_SIZE) = 0;


        // Interface fetching for logic
        virtual GPUBaseAcceleratorI&                BaseAcelerator() = 0;
        virtual const AcceleratorBatchMappings&     AcceleratorBatches() = 0;
        virtual const MaterialBatchMappings&        MaterialBatches() = 0;
        virtual const AcceleratorGroupList&         AcceleratorGroups() = 0;
        virtual const MaterialGroupList&            MaterialGroups() = 0;

        // Returns max bits of keys (for batch and id respectively)
        virtual const Vector2i                      SceneMaterialMaxBits() const = 0;
        virtual const Vector2i                      SceneAcceleratorMaxBits() const = 0;

        // Options of the Hitman & Shademan
        virtual const HitOpts&                      HitOptions() const = 0;
        virtual const ShadeOpts&                    ShadeOptions() const = 0;

        // Misc
        // Retuns "sizeof(RayAux)"
        virtual size_t                              PerRayAuxDataSize() const = 0;
        // Return mimimum size of an arbitrary struct which holds all hit results
        virtual size_t                              HitStructSize() const = 0;
        // Random seed
        virtual uint32_t                            Seed() const = 0;
};