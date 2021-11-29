#pragma once

#include "Matrix.h"
#include "HitStructs.h"

#include "TracerStructs.h"

#include "RayLib/Flag.h"

struct SceneError;
struct TextureStruct;
struct SceneAnalyticData;

enum class SceneLoadFlagType
{
    FORCE_OPTIX_ACCELS,
    END
};

using SceneLoadFlags = Flags<SceneLoadFlagType>;

class GPUSceneI
{
    public:
        virtual                 ~GPUSceneI() = default;

        // Interface
        virtual size_t          UsedGPUMemory() const = 0;
        virtual size_t          UsedCPUMemory() const = 0;
        //
        virtual SceneError      LoadScene(double) = 0;
        virtual SceneError      ChangeTime(double) = 0;
        //
        virtual Vector2i        MaxMatIds() const  = 0;
        virtual Vector2i        MaxAccelIds() const  = 0;
        virtual HitKey          BaseBoundaryMaterial() const = 0;
        virtual uint32_t        HitStructUnionSize() const = 0;
        virtual double          MaxSceneTime() const = 0;
        virtual uint32_t        CameraCount() const = 0;
        // Access CPU
        virtual const NamedList<CPULightGPtr>&      Lights() const = 0;
        virtual const NamedList<CPUCameraGPtr>&     Cameras() const = 0;
        virtual const NamedList<CPUTransformGPtr>&  Transforms() const = 0;
        virtual const NamedList<CPUMediumGPtr>&     Mediums() const = 0;
        //
        virtual uint16_t                            BaseMediumIndex() const = 0;
        virtual uint32_t                            IdentityTransformIndex() const = 0;
        virtual uint32_t                            BoundaryTransformIndex() const = 0;
        // Generated Classes of Materials / Accelerators
        // Work Maps
        virtual const WorkBatchCreationInfo&            WorkBatchInfo() const = 0;
        virtual const BoundaryWorkBatchCreationInfo&    BoundarWorkBatchInfo() const = 0;
        virtual const AcceleratorBatchMap&              AcceleratorBatchMappings() const = 0;
        // Allocated Types
        // All of which are allocated on the GPU
        virtual const GPUBaseAccelPtr&                      BaseAccelerator() const = 0;
        virtual const std::map<NameGPUPair, GPUMatGPtr>&    MaterialGroups() const = 0;
        virtual const NamedList<GPUAccelGPtr>&              AcceleratorGroups() const = 0;
        virtual const NamedList<GPUPrimGPtr>&               PrimitiveGroups() const = 0;
        // Analytic Data Generation
        virtual SceneAnalyticData                           AnalyticData() const = 0;
};