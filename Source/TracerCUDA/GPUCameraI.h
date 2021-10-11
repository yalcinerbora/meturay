#pragma once

#include "GPUEndpointI.h"

#include "RayLib/Constants.h"
#include "RayLib/SceneStructs.h"

#include "CudaSystem.h"
#include "Random.cuh"
#include "RayStructs.h"
#include "NodeListing.h"

class GPUTransformI;
class RandomGPU;

struct VisorCamera;

class GPUCameraPixel;

class GPUCameraI : public GPUEndpointI
{
    public:
        __device__              GPUCameraI(uint16_t mediumIndex,
                                           HitKey, TransformId,
                                           const GPUTransformI&,
                                           PrimitiveId = 0);
        virtual                 ~GPUCameraI() = default;

        // Interface
        __device__
        virtual uint32_t        FindPixelId(const RayReg& r,
                                            const Vector2i& resolution) const = 0;
        // View Projection Matrix of the Camera
        // If camera projection is linearly transformable 
        // else it returns only the view matrix
        __device__
        virtual Matrix4x4       VPMatrix() const = 0;
        // Distance of near and far planes of the Camera
        // in world space distances
        __device__
        virtual Vector2f        NearFar() const = 0;

        __device__
        virtual GPUCameraPixel  GeneratePixelCamera(const Vector2i& pixelId,
                                                    const Vector2i& resolution) const = 0;
};

using GPUCameraList = std::vector<const GPUCameraI*>;
using VisorCameraList = std::vector<VisorCamera>;

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
        virtual uint32_t					    CameraCount() const = 0;

        virtual size_t						UsedGPUMemory() const = 0;
        virtual size_t						UsedCPUMemory() const = 0;
};

__device__
inline GPUCameraI::GPUCameraI(uint16_t mediumIndex,
                              HitKey hK, TransformId tId,
                              const GPUTransformI& gTrans,
                              PrimitiveId pId)
    : GPUEndpointI(mediumIndex, hK, tId, pId, gTrans)
{}