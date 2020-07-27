#pragma once

#include <cstdint>

#include "RayLib/HitStructs.h"
#include "RayLib/Vector.h"

struct RayGMem;
struct SceneError;

class RNGMemory;
class GPUPrimitiveGroupI;
class GPUMaterialGroupI;

// Defines call action over a certain material group/ primitive group
// pair
// These batches defined by renderer(integrator)

// Work Batches are responsible for creating surfaces required by the material
// and any material previous work
class GPUWorkBatchI
{
    public:
        virtual                             ~GPUWorkBatchI() = default;

        // Interface
        // Type (as string) of the primitive group
        virtual const char*                 Type() const = 0;
        // Get ready for kernel call
        virtual void                        GetReady() = 0;
        // Kernel Call
        virtual void                        Work(// Output
                                                 HitKey* dBoundMatOut,
                                                 RayGMem* dRayOut,
                                                 //  Input
                                                 const RayGMem* dRayIn,
                                                 const PrimitiveId* dPrimitiveIds,
                                                 const HitStructPtr dHitStructs,
                                                 // Ids
                                                 const HitKey* dMatIds,
                                                 const RayId* dRayIds,
                                                 //
                                                 const uint32_t rayCount,
                                                 RNGMemory& rngMem) = 0;

        // Every MaterialBatch is available for a specific primitive / material data
        virtual const GPUPrimitiveGroupI&   PrimitiveGroup() const = 0;
        virtual const GPUMaterialGroupI&    MaterialGroup() const = 0;

        virtual uint8_t                     OutRayCount() const = 0;
};

