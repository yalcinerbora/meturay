#pragma once

#include "GPUEndpointI.h"

#include "RayLib/Constants.h"
#include "RayLib/SceneStructs.h"

#include "CudaConstants.h"
#include "Random.cuh"
#include "RayStructs.h"
#include "NodeListing.h"

class GPUTransformI;
class RandomGPU;

using GPUCameraI = GPUEndpointI;
using GPUCameraList = std::vector<const GPUCameraI*>;

class CPUCameraGroupI
{
    public:
        virtual								~CPUCameraGroupI() = default;
    
        // Interface
        virtual const char*                 Type() const = 0;
        virtual const GPUCameraList&        GPUCameras() const = 0;
        virtual SceneError					InitializeGroup(const CameraGroupData& cameraNodes,
                                                            const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                            const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                            const MaterialKeyListing& allMaterialKeys,
                                                            double time,
                                                            const std::string& scenePath) = 0;
        virtual SceneError					ChangeTime(const NodeListing& lightNodes, double time,
                                                       const std::string& scenePath) = 0;
        virtual TracerError					ConstructCameras(const CudaSystem&) = 0;
        virtual uint32_t					CameraCount() const = 0;
    
        virtual size_t						UsedGPUMemory() const = 0;
        virtual size_t						UsedCPUMemory() const = 0;
    
        virtual void						AttachGlobalTransformArray(const GPUTransformI** deviceTranfsorms) = 0;
};