#pragma once

#include "Matrix.h"
#include "HitStructs.h"

struct SceneError;
struct LightStruct;
struct CameraPerspective;
struct TracerParameters;

using TransformStruct = Matrix4x4;

class GPUSceneI
{
    public:
        virtual                             ~GPUSceneI() = default;

        // Interface
        virtual size_t                      UsedGPUMemory() = 0;
        virtual size_t                      UsedCPUMemory() = 0;
        //
        virtual SceneError                  LoadScene(const TracerParameters&, double) = 0;
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
};