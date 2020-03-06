#pragma once

#include "Matrix.h"
#include "HitStructs.h"

#include "TracerStructs.h"

struct SceneError;
struct LightStruct;
struct CameraPerspective;
struct TracerParameters;

using TransformStruct = Matrix4x4;

using MatPrimPair = std::pair<const GPUPrimitiveGroupI, const GPUMaterialGroupI>;
using WorkBatchMappings = std::map<HitKey, MatPrimPair>;

class GPUSceneI
{
    public:
        virtual                             ~GPUSceneI() = default;

        // Interface
        virtual size_t                      UsedGPUMemory() = 0;
        virtual size_t                      UsedCPUMemory() = 0;
        //
        virtual SceneError                  LoadScene(double) = 0;
        virtual SceneError                  ChangeTime(double) = 0;
        //
        virtual Vector2i                    MaxMatIds() = 0;
        virtual Vector2i                    MaxAccelIds() = 0;
        virtual HitKey                      BaseBoundaryMaterial() = 0;
        // Access GPU
        virtual const LightStruct*          LightsGPU() const = 0;
        virtual const TransformStruct*      TransformsGPU() const = 0;
        // Access CPU
        virtual const CameraPerspective*    CamerasCPU() const = 0;

        // Generated Classes of Materials / Accelerators
        // Work Maps
        virtual const WorkBatchMappings&            WorkBatchMap() const = 0;
        virtual const AcceleratorBatchMappings&     AcceleratorBatchMappings() const = 0;

        // Allocated Types
        // All of which are allocated on the GPU
        virtual const GPUBaseAccelPtr&                          BaseAccelerator() const = 0;
        virtual const std::map<NameGPUPair, GPUMatGPtr>&        MaterialGroups() const = 0;
        virtual const std::map<std::string, GPUAccelGPtr>&      AcceleratorGroups() const = 0;
        virtual const std::map<std::string, GPUPrimGPtr>&       PrimitiveGroups() const = 0;
};