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

        // TODO: Change this later
        // This is a weird interface but it is simpler to implement
        // Given a memory region construct the child class over the region
        // Return the interface as parent class.
        // Interface is like this because, a kernel want to construct a sub camera
        // over a shared memory without knowing the underlying type (or register space
        // local memory).
        __device__
        virtual GPUCameraI*     GenerateSubCamera(Byte* memoryRegion, size_t size,
                                                  const Vector2i& regionId,
                                                  const Vector2i& regionCount) const = 0;


        virtual __device__ void     Test(// Output
                                 RayReg& ray,
                                 Vector2f& localCoords,
                                 // Input,
                                 const Vector2i& sampleIdInner,
                                 const Vector2i& sampleIdOuter,
                                 const Vector2i& sampleCountInner,
                                 const Vector2i& sampleCountOuter,
                                 // I-O
                                 RNGeneratorGPUI& rng,
                                 // Options
                                 bool antiAliasOn) const = 0;
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