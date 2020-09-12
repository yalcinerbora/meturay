#pragma once

#include "Matrix.h"
#include "HitStructs.h"

#include "TracerStructs.h"

struct SceneError;
struct CPULight;
struct CPUCamera;
struct CPUMedium;
struct CPUTransform;
struct TextureStruct;

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
        // Access CPU                
        virtual const std::vector<CPULight>&        LightsCPU() const = 0;
        virtual const std::vector<CPUCamera>&       CamerasCPU() const = 0;
        virtual const NamedList<CPUTransformGPtr>&  Transforms() const = 0;
        virtual const NamedList<CPUMediumGPtr>&     Mediums() const = 0;
        //
        virtual uint32_t                            BaseMediumIndex() const = 0;
        virtual uint32_t                            IdentityTransformIndex() const = 0;
        // Generated Classes of Materials / Accelerators
        // Work Maps
        virtual const WorkBatchCreationInfo&        WorkBatchInfo() const = 0;
        virtual const AcceleratorBatchMap&          AcceleratorBatchMappings() const = 0;
        // Allocated Types
        // All of which are allocated on the GPU
        virtual const GPUBaseAccelPtr&                      BaseAccelerator() const = 0;
        virtual const std::map<NameGPUPair, GPUMatGPtr>&    MaterialGroups() const = 0;
        virtual const NamedList<GPUAccelGPtr>&              AcceleratorGroups() const = 0;
        virtual const NamedList<GPUPrimGPtr>&               PrimitiveGroups() const = 0;
};