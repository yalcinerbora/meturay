#pragma once

#include <cstdint>

#include "RayLib/HitStructs.h"
#include "RayLib/Vector.h"

struct RayGMem;
struct SceneError;

class RNGeneratorCPUI;
class GPUPrimitiveGroupI;
class GPUMaterialGroupI;
class CPUEndpointGroupI;

// Defines call action over a certain material group/ primitive group
// pair
// These batches defined by renderer(integrator)

// Work Batches are responsible for creating surfaces required by the material
// and any material previous work
class GPUWorkBatchI
{
    public:
        virtual                 ~GPUWorkBatchI() = default;

        // Interface
        // Type (as string) of the primitive group
        virtual const char*     Type() const = 0;
        // Get ready for kernel call
        virtual void            GetReady() = 0;
        // Kernel Call
        virtual void            Work(// Output
                                     HitKey* dBoundMatOut,
                                     RayGMem* dRayOut,
                                     //  Input
                                     const RayGMem* dRayIn,
                                     const PrimitiveId* dPrimitiveIds,
                                     const TransformId* dTransformIds,
                                     const HitStructPtr dHitStructs,
                                     // Ids
                                     const HitKey* dMatIds,
                                     const RayId* dRayIds,
                                     //
                                     const uint32_t rayCount,
                                     RNGeneratorCPUI& rngCPU) = 0;

        virtual uint8_t         OutRayCount() const = 0;
};
