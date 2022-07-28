#pragma once

#include "GPUEndpointI.h"

#include "RayLib/Constants.h"
#include "RayLib/SceneStructs.h"

#include "CudaSystem.h"
#include "RayStructs.h"
#include "NodeListing.h"

struct VisorTransform;
class GPUTransformI;
class GPUCameraI;
class GPUCameraPixel;
class DeviceMemory;

using GPUCameraList = std::vector<const GPUCameraI*>;

class GPUCameraI : public GPUEndpointI
{
    public:
        __device__              GPUCameraI(uint16_t mediumIndex,
                                           HitKey, const GPUTransformI&);
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
        // Field of View of the camera
        __device__
        virtual Vector2f        FoV() const = 0;

        __device__
        virtual VisorTransform  GenVisorTransform() const = 0;
        __device__
        virtual void            SwapTransform(const VisorTransform&) = 0;

        __device__
        virtual GPUCameraPixel  GeneratePixelCamera(const Vector2i& pixelId,
                                                    const Vector2i& resolution) const = 0;
};

class CPUCameraGroupI : public CPUEndpointGroupI
{
    public:
        virtual							~CPUCameraGroupI() = default;

        // Interface
        virtual const GPUCameraList&    GPUCameras() const = 0;

        virtual void                    CopyCamera(DeviceMemory&,
                                                   const GPUCameraI* gCamera,
                                                   const CudaSystem& cudaSystem) = 0;
};

__device__
inline GPUCameraI::GPUCameraI(uint16_t mediumIndex, HitKey hk,
                              const GPUTransformI& gTrans)
    : GPUEndpointI(mediumIndex, hk, gTrans)
{}