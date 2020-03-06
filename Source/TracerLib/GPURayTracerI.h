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
#include "RayLib/HitStructs.h"

//struct TracerError;
//class GPUBaseAcceleratorI;
//
//class GPURayTracerI
//{
//    public:
//        virtual                                     ~GPURayTracerI() = default;
//
//        // Interface
//        // Initialize and allocate for rays
//        virtual TracerError                         Initialize() = 0;
//        virtual void                                ResetRayMemory(uint32_t rayCount) = 0;
//
//        virtual void                                HitRays() = 0;
//        virtual void                                WorkRays(const WorkBatchMappings) = 0;
//
//        // Interface fetching for logic     
//        virtual const GPUBaseAcceleratorI           BaseAccelerator() = 0;
//        virtual const AcceleratorBatchMappings&     AcceleratorBatches() = 0;
//        virtual const WorkBatchMappings&            CurrentWorkBatchLogic() = 0;
//
//        // Returns max bits of keys (for batch and id respectively)
//        virtual const Vector2i                      SceneMaterialMaxBits() const = 0;
//        virtual const Vector2i                      SceneAcceleratorMaxBits() const = 0;
//
//        virtual const HitKey                        SceneBaseBoundMatKey() const = 0;
//        virtual const TracerParameters&             Parameters() const = 0;
//
//        // Return mimimum size of an arbitrary struct which holds all hit results
//        virtual size_t                              HitStructSize() const = 0;
//};