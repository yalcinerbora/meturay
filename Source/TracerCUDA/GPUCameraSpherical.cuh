﻿#pragma once

#include "GPUCameraI.h"
#include "DeviceMemory.h"
#include "GPUTransformI.h"
#include "TypeTraits.h"
#include "GPUCameraPixel.cuh"

#include "RayLib/VisorCamera.h"
#include "RayLib/CoordinateConversion.h"

class GPUCameraSpherical final : public GPUCameraI
{
    private:
        float                   pixelRatio;
        Vector3                 position;
        Vector3                 direction;
        Vector3                 right;
        Vector3                 up;
        Vector2                 nearFar;

    protected:
    public:
        // Constructors & Destructor
        __device__          GPUCameraSpherical(float pixelRatio,
                                               const Vector3& position,
                                               const Vector3& direction,
                                               const Vector3& up,
                                               const Vector2& nearFar,
                                               const GPUTransformI& transform,
                                               //
                                               uint16_t mediumId,
                                               HitKey materialKey);
                            ~GPUCameraSpherical() = default;

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

        __device__ GPUCameraPixel   GeneratePixelCamera(const Vector2i& pixelId,
                                                        const Vector2i& resolution) const override;
};

class CPUCameraGroupSpherical final : public CPUCameraGroupI
{
    public:
        static const char* TypeName() { return "Spherical"; }
        // Node Names
        static constexpr const char* NAME_POSITION  = "position";
        static constexpr const char* NAME_DIR       = "direction";
        static constexpr const char* NAME_UP        = "up";
        static constexpr const char* NAME_PLANES    = "planes";
        static constexpr const char* NAME_PIX_RATIO = "pixelRatio";

        struct Data
        {
            // Sample Ready Parameters
            // All of which is world space
            float                   pixelRatio;
            Vector3                 position;
            Vector3                 direction;
            Vector3                 up;
            Vector2                 nearFar;
        };

        //using Data = GPUCameraPinhole::Data;

    private:
        DeviceMemory                    memory;
        const GPUCameraSpherical*       dGPUCameras;

        std::vector<HitKey>             hHitKeys;
        std::vector<uint16_t>           hMediumIds;
        std::vector<TransformId>        hTransformIds;
        std::vector<Data>               hCameraData;

        GPUCameraList                   gpuCameraList;
        VisorCameraList                 visorCameraList;
        uint32_t                        cameraCount;

    protected:
    public:
        // Constructors & Destructor
                                        CPUCameraGroupSpherical();
                                        ~CPUCameraGroupSpherical() = default;

        // Interface
        const char*                     Type() const override;
        const GPUCameraList&            GPUCameras() const override;
        const VisorCameraList&          VisorCameras() const override;
        SceneError					    InitializeGroup(const CameraGroupDataList& cameraNodes,
                                                        const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                        const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                        uint32_t cameraMaterialBatchId,
                                                        double time,
                                                        const std::string& scenePath) override;
        SceneError					    ChangeTime(const NodeListing& lightNodes, double time,
                                                   const std::string& scenePath) override;
        TracerError					    ConstructCameras(const CudaSystem&,
                                                         const GPUTransformI**) override;
        uint32_t					        CameraCount() const override;

        size_t						    UsedGPUMemory() const override;
        size_t						    UsedCPUMemory() const override;
};

__device__
inline GPUCameraSpherical::GPUCameraSpherical(float pixelRatio,
                                              const Vector3& pos,
                                              const Vector3& dir,
                                              const Vector3& upp,
                                              const Vector2& nearFar,
                                              const GPUTransformI& transform,
                                              //
                                              uint16_t mediumId,
                                              HitKey materialKey)
    : GPUCameraI(materialKey, mediumId)
    , position(transform.LocalToWorld(pos))
    , up(transform.LocalToWorld(upp))
    , direction(transform.LocalToWorld(dir))
    , nearFar(nearFar)
    , pixelRatio(pixelRatio)
{    
    // Camera Vector Correction    
    right = Cross(direction, up).Normalize();
    up = Cross(right, direction).Normalize();
    direction = Cross(up, right).Normalize();
}

__device__
inline void GPUCameraSpherical::Sample(// Output
                                       float& distance,
                                       Vector3& dirOut,
                                       float& pdf,
                                       // Input
                                       const Vector3& sampleLoc,
                                       // I-O
                                       RandomGPU&) const
{
    // One
    dirOut = sampleLoc - position;
    distance = dirOut.Length();
    dirOut.NormalizeSelf();
    pdf = 1.0f;
}

__device__
inline void GPUCameraSpherical::GenerateRay(// Output
                                            RayReg& ray,
                                            // Input
                                            const Vector2i& sampleId,
                                            const Vector2i& sampleMax,
                                            // I-O
                                            RandomGPU& rng,
                                            // Options
                                            bool antiAliasOn) const
{
    //if(threadIdx.x == 0)
    //{
    //    printf("SI: %d, %d | SM: %d, %d\n", 
    //           sampleId[0], sampleMax[1],
    //           sampleMax[0], sampleMax[1]);
    //}
   
    // Create random location over sample pixel
    Vector2 randomOffset = (antiAliasOn)
                                ? Vector2(GPUDistribution::Uniform<float>(rng),
                                          GPUDistribution::Uniform<float>(rng))
                                : Vector2(0.5f);

    // Normalize Coordinates X & Y = [0, 1]
    Vector2f normCoords = Vector2(static_cast<float>(sampleId[0]),
                                  static_cast<float>(sampleId[1]));
    normCoords += randomOffset;
    normCoords /= Vector2(static_cast<float>(sampleMax[0]),
                          static_cast<float>(sampleMax[1]));

    // Calculate Spherical Coordinates
    Vector2f sphericalCoords = Vector2f(// [-pi, pi]
                                        (normCoords[0] * MathConstants::Pi * 2.0f) - MathConstants::Pi,
                                        // [0, pi]
                                        (1.0f - normCoords[1]) * MathConstants::Pi);

    // Incorporate pixel ratio
    // TODO: Incorporate Pix Ratio (for non-square textures)
    sphericalCoords[0] *= pixelRatio;
    
    // Convert to Cartesian
    Vector3 dirZUp = Utility::SphericalToCartesianUnit(sphericalCoords);
    Vector3 dirYUp = Vector3(dirZUp[1], dirZUp[2], dirZUp[0]);

    // Finally transform from local to world
    Matrix3x3 viewMat;
    TransformGen::Space(viewMat, right, up, direction);
    dirYUp = viewMat * dirYUp;
    
    // DEBUG
    //printf("NC: (%f, %f), DIR: (%f, %f, %f)\n",
    //       normCoords[0], normCoords[1],
    //       dirYUp[0], dirYUp[1], dirYUp[2]);
    //printf("%f, %f\n", sphericalCoords[0], sphericalCoords[1]);

    // Initialize Ray
    ray.ray = RayF(dirYUp, position);
    ray.tMin = nearFar[0];
    ray.tMax = nearFar[1];
}

__device__
inline float GPUCameraSpherical::Pdf(const Vector3& worldDir,
                                     const Vector3& worldPos) const
{
    return 0.0f;
}

__device__
inline uint32_t GPUCameraSpherical::FindPixelId(const RayReg& r,
                                                const Vector2i& resolution) const
{
    // TODO:
    return 0;
}

inline __device__ bool GPUCameraSpherical::CanBeSampled() const
{
    return false;
}

__device__
inline PrimitiveId GPUCameraSpherical::PrimitiveIndex() const
{
    return 0;
}

__device__ 
inline Matrix4x4 GPUCameraSpherical::VPMatrix() const
{
    // Well we cant generate Projection Matrix (or a Matrix)
    // for cartesian => spherical only return view matrix
    Matrix3x3 viewMatrix;
    TransformGen::Space(viewMatrix, right, up, direction);
    return ToMatrix4x4(viewMatrix);
}

__device__ 
inline Vector2f GPUCameraSpherical::NearFar() const
{
    return nearFar;
}

__device__
inline GPUCameraPixel GPUCameraSpherical::GeneratePixelCamera(const Vector2i& pixelId,
                                                              const Vector2i& resolution) const
{
    const auto ZERO3 = Zero3;
    const auto ZERO2 = Zero2;
    // TODO: Implement
    return GPUCameraPixel(ZERO3,
                          ZERO3,
                          ZERO3,
                          ZERO3,
                          ZERO2,
                          ZERO3,
                          pixelId,
                          resolution,
                          mediumIndex,
                          boundaryMaterialKey);
}

inline CPUCameraGroupSpherical::CPUCameraGroupSpherical()
    : dGPUCameras(nullptr)
    , cameraCount(0)
{}

inline const char* CPUCameraGroupSpherical::Type() const
{
    return TypeName();
}

inline const GPUCameraList& CPUCameraGroupSpherical::GPUCameras() const
{
    return gpuCameraList;
}

inline const VisorCameraList& CPUCameraGroupSpherical::VisorCameras() const
{
    return visorCameraList;
}

inline uint32_t CPUCameraGroupSpherical::CameraCount() const
{
    return cameraCount;
}

inline size_t CPUCameraGroupSpherical::UsedGPUMemory() const
{
    return memory.Size();
}

inline size_t CPUCameraGroupSpherical::UsedCPUMemory() const
{
    return (sizeof(HitKey) * hHitKeys.size() +
            sizeof(uint16_t) * hMediumIds.size() +
            sizeof(TransformId) * hTransformIds.size() +
            sizeof(Data) * hCameraData.size());
}

static_assert(IsTracerClass<CPUCameraGroupSpherical>::value,
              "CPUCameraSpherical is not a tracer class");