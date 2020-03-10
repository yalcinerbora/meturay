#pragma once

#include "Matrix.h"
#include "HitStructs.h"

#include "TracerStructs.h"

struct SceneError;
class GPULightI;
class GPUEndpointI;
using GPUCameraI = GPUEndpointI;

using TransformStruct = Matrix4x4;

class GPUSceneI
{
    public:
        virtual                             ~GPUSceneI() = default;

        // Interface
        virtual size_t                      UsedGPUMemory() const = 0;
        virtual size_t                      UsedCPUMemory() const = 0;
        //
        virtual SceneError                  LoadScene(double) = 0;
        virtual SceneError                  ChangeTime(double) = 0;
        //
        virtual Vector2i                    MaxMatIds() const  = 0;
        virtual Vector2i                    MaxAccelIds() const  = 0;
        virtual HitKey                      BaseBoundaryMaterial() const = 0;
        virtual uint32_t                    HitStructUnionSize() const = 0;
        // Access GPU                
        virtual const GPULightI*            LightsGPU() const = 0;
        virtual const GPUCameraI*           CamerasGPU() const = 0;
        virtual const TransformStruct*      TransformsGPU() const = 0;
        // Access CPU
        
        // Counts
        virtual size_t                      LightCount() const = 0;
        virtual size_t                      TransformCount() const = 0;
        virtual size_t                      CameraCount() const = 0;
        
        // Generated Classes of Materials / Accelerators
        // Work Maps
        virtual const WorkBatchCreationInfo&    WorkBatchInfo() const = 0;
        virtual const AcceleratorBatchMap&      AcceleratorBatchMappings() const = 0;
        // Allocated Types
        // All of which are allocated on the GPU
        virtual const GPUBaseAccelPtr&                          BaseAccelerator() const = 0;
        virtual const std::map<NameGPUPair, GPUMatGPtr>&        MaterialGroups() const = 0;
        virtual const std::map<std::string, GPUAccelGPtr>&      AcceleratorGroups() const = 0;
        virtual const std::map<std::string, GPUPrimGPtr>&       PrimitiveGroups() const = 0;
};