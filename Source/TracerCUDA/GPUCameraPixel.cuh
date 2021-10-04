#pragma once

#include "GPUCameraI.h"
#include "DeviceMemory.h"
#include "GPUTransformI.h"
#include "TypeTraits.h"
#include "GPUCameraPinhole.cuh"

#include "RayLib/VisorCamera.h"

class GPUCameraPixel final : public GPUCameraI
{
    private:
        const GPUCameraI&       baseCamera;
        // Pixel Location Related
        int32_t                 pixelId;
        Vector2i                resolution;

    protected:
    public:
        // Constructors & Destructor
        __device__          GPUCameraPixel(const GPUCameraI& baseCamera,
                                           int32_t pixelId,
                                           const Vector2i& resolution);
                            ~GPUCameraPixel() = default;

        // Interface
        __device__ void     Sample(// Output
                                   float& distance,
                                   Vector3& direction,
                                   float& pdf,
                                   // Input
                                   const Vector3& position,
                                   // I-O
                                   RandomGPU&) const override;

        __device__ void     GenerateRay(// Output
                                        RayReg&,
                                        // Input
                                        const Vector2i& sampleId,
                                        const Vector2i& sampleMax,
                                        // I-O
                                        RandomGPU&,
                                        // Options
                                        bool antiAliasOn) const override;
        __device__ float    Pdf(const Vector3& direction,
                                const Vector3& position) const override;


        __device__ uint32_t         FindPixelId(const RayReg& r,
                                               const Vector2i& resolution) const override;

        __device__ bool             CanBeSampled() const override;
        __device__ PrimitiveId      PrimitiveIndex() const override;

        __device__ Matrix4x4        VPMatrix() const override;
        __device__ Vector2f         NearFar() const override;
};

__device__
inline GPUCameraPixel::GPUCameraPixel(const GPUCameraI& baseCamera,
                                      int32_t pixelId,
                                      const Vector2i& resolution)
    : GPUCameraI(baseCamera.BoundaryMaterial(), baseCamera.MediumIndex())
    , baseCamera(baseCamera)
    , pixelId(pixelId)
    , resolution(resolution)
{}

__device__
inline void GPUCameraPixel::Sample(// Output
                                   float& distance,
                                   Vector3& direction,
                                   float& pdf,
                                   // Input
                                   const Vector3& sampleLoc,
                                   // I-O
                                   RandomGPU& rng) const
{
    // TODO: fix this
    baseCamera.Sample(distance, direction, pdf, sampleLoc, rng);
}

__device__
inline void GPUCameraPixel::GenerateRay(// Output
                                        RayReg& ray,
                                        // Input
                                        const Vector2i& sampleId,
                                        const Vector2i& sampleMax,
                                        // I-O
                                        RandomGPU& rng,
                                        // Options
                                        bool antiAliasOn) const
{
    
}

__device__
inline float GPUCameraPixel::Pdf(const Vector3& worldDir,
                                 const Vector3& worldPos) const
{
    // TODO: Change this
    return baseCamera.Pdf(worldDir, worldPos);
}

__device__
inline uint32_t GPUCameraPixel::FindPixelId(const RayReg& r,
                                            const Vector2i& resolution) const
{
    uint32_t pixelId = baseCamera.FindPixelId(r, resolution);
    // TODO change this
    return pixelId;
}

inline __device__ bool GPUCameraPixel::CanBeSampled() const
{
    return baseCamera.CanBeSampled();
}

__device__
inline PrimitiveId GPUCameraPixel::PrimitiveIndex() const
{
    return baseCamera.PrimitiveIndex();
}

__device__ 
inline Matrix4x4 GPUCameraPixel::VPMatrix() const
{
    return baseCamera.VPMatrix();
}

__device__ 
inline Vector2f GPUCameraPixel::NearFar() const
{
    return baseCamera.NearFar();
}

__global__
void KCConstructSingleGPUCameraPixel(GPUCameraPixel* gCameraLocations,
                                     bool deletePrevious,
                                     //
                                     const GPUCameraI& baseCam,
                                     int32_t pixelIndex,
                                     Vector2i resolution);