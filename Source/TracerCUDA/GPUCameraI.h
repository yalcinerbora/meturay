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

struct VisorCamera;

class GPUCameraI : public GPUEndpointI
{
    public:
        __device__              GPUCameraI(HitKey k, uint16_t mediumIndex);
        virtual                 ~GPUCameraI() = default;

        // Interface
        __device__
        virtual uint32_t        FindPixelId(const RayReg& r,
                                            const Vector2i& resolution) const = 0;
};

using GPUCameraList = std::vector<const GPUCameraI*>;
using VisorCameraList = std::vector<const VisorCamera>;

__device__
inline GPUCameraI::GPUCameraI(HitKey k, uint16_t mediumIndex)
    : GPUEndpointI(k, mediumIndex)
{}

class CPUCameraGroupI
{
    public:
        virtual								~CPUCameraGroupI() = default;

        // Interface
        virtual const char*                 Type() const = 0;
        virtual const GPUCameraList&        GPUCameras() const = 0;
        virtual const VisorCameraList&      VisorCameras() const = 0;
        virtual SceneError					InitializeGroup(const CameraGroupDataList& cameraNodes,
                                                            const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                            const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                            uint32_t cameraMaterialBatchId,
                                                            double time,
                                                            const std::string& scenePath) = 0;
        virtual SceneError					ChangeTime(const NodeListing& cameraNodes, double time,
                                                       const std::string& scenePath) = 0;
        virtual TracerError					ConstructCameras(const CudaSystem&,
                                                             const GPUTransformI**) = 0;
        virtual uint32_t					CameraCount() const = 0;
        virtual VisorCamera                 GetCameraAsVisorCam(uint32_t camId) = 0;

        virtual size_t						UsedGPUMemory() const = 0;
        virtual size_t						UsedCPUMemory() const = 0;
};