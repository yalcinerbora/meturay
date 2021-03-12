#pragma once

#include "GPUCameraI.h"
#include "DeviceMemory.h"
#include "GPUTransformI.h"
#include "TypeTraits.h"

class GPUCameraPinhole final : public GPUCameraI
{
    private:
        Vector3                 position;
        Vector3                 right;
        Vector3                 up;
        Vector3                 bottomLeft;
        Vector2                 planeSize;
        Vector2                 nearFar;

    protected:
    public:
        // Constructors & Destructor
        __device__          GPUCameraPinhole(const Vector3& position,
                                             const Vector3& gaze,
                                             const Vector3& up,
                                             const Vector2& nearFar,
                                             const Vector2& fov,
                                             const GPUTransformI& transform,
                                             //
                                             uint16_t mediumId,
                                             HitKey materialKey);
                            ~GPUCameraPinhole() = default;

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
                                        RandomGPU&) const override;

        __device__ PrimitiveId      PrimitiveIndex() const override;
};

class CPUCameraGroupPinhole final : public CPUCameraGroupI
{
    public:
        static const char* TypeName() { return "Pinhole"; }
        // Node Names
        static constexpr const char* NAME_POSITION  = "position";
        static constexpr const char* NAME_GAZE      = "gaze";
        static constexpr const char* NAME_UP        = "up";
        static constexpr const char* NAME_PLANES    = "planes";
        static constexpr const char* NAME_FOV       = "fov";

        struct Data
        {
            // Sample Ready Parameters
            // All of which is world space
            Vector3                 position;
            Vector3                 gaze;
            Vector3                 up;
            Vector2                 nearFar;
            Vector2                 fov;
        };



        //using Data = GPUCameraPinhole::Data;

    private:
        DeviceMemory                    memory;
        const GPUCameraPinhole*         dGPUCameras;

        std::vector<HitKey>             hHitKeys;
        std::vector<uint16_t>           hMediumIds;
        std::vector<TransformId>        hTransformIds;
        std::vector<Data>               hCameraData;

        GPUCameraList                   gpuCameraList;
        uint32_t                        cameraCount;

    protected:
    public:
        // Constructors & Destructor
                                        CPUCameraGroupPinhole();
                                        ~CPUCameraGroupPinhole() = default;

        // Interface
        const char*                     Type() const override;
        const GPUCameraList&            GPUCameras() const override;
        SceneError					    InitializeGroup(const CameraGroupData& cameraNodes,
                                                        const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                        const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                        const MaterialKeyListing& allMaterialKeys,
                                                        double time,
                                                        const std::string& scenePath) override;
        SceneError					    ChangeTime(const NodeListing& lightNodes, double time,
                                                   const std::string& scenePath) override;
        TracerError					    ConstructCameras(const CudaSystem&,
                                                         const GPUTransformI**) override;
        uint32_t					    CameraCount() const override;

        size_t						    UsedGPUMemory() const override;
        size_t						    UsedCPUMemory() const override;
};

__device__
inline GPUCameraPinhole::GPUCameraPinhole(const Vector3& pos,
                                          const Vector3& gz,
                                          const Vector3& upp,
                                          const Vector2& nearFar,
                                          const Vector2& fov,
                                          const GPUTransformI& transform,
                                          //
                                          uint16_t mediumId,
                                          HitKey materialKey)
    : GPUEndpointI(materialKey, mediumId)
    , position(transform.LocalToWorld(pos))
    , up(transform.LocalToWorld(upp))
    , planeSize(planeSize)
    , nearFar(nearFar)
{

    Vector3 gazePoint = transform.LocalToWorld(gz);
    float nearPlane = nearFar[0];
    float farPLane = nearFar[1];

    // Find world space window sizes
    float widthHalf = tanf(fov[0] * 0.5f) * nearPlane;
    float heightHalf = tanf(fov[1] * 0.5f) * nearPlane;

    // Camera Vector Correction
    Vector3 gazeDir = gazePoint - position;
    right = Cross(gazeDir, up).Normalize();
    up = Cross(right, gazeDir).Normalize();
    gazeDir = Cross(up, right).Normalize();

    // Camera parameters
    bottomLeft = (position
                  - right * widthHalf
                  - up * heightHalf
                  + gazeDir * nearPlane);

    planeSize = Vector2(widthHalf, heightHalf) * 2.0f;
}

__device__
inline void GPUCameraPinhole::Sample(// Output
                                     float& distance,
                                     Vector3& direction,
                                     float& pdf,
                                     // Input
                                     const Vector3& sampleLoc,
                                     // I-O
                                     RandomGPU&) const
{
    // One
    direction = sampleLoc - position;
    distance = direction.Length();
    direction.NormalizeSelf();
    pdf = 1.0f;
}

__device__
inline void GPUCameraPinhole::GenerateRay(// Output
                                          RayReg& ray,
                                          // Input
                                          const Vector2i& sampleId,
                                          const Vector2i& sampleMax,
                                          // I-O
                                          RandomGPU& rng) const
{
    // DX DY from stratfied sample
    Vector2 delta = Vector2(planeSize[0] / static_cast<float>(sampleMax[0]),
                            planeSize[1] / static_cast<float>(sampleMax[1]));

    // Create random location over sample rectangle
    float dX = GPUDistribution::Uniform<float>(rng);
    float dY = GPUDistribution::Uniform<float>(rng);
    Vector2 randomOffset = Vector2(dX, dY);
    //Vector2 randomOffset = Vector2(0.5f);

    Vector2 sampleDistance = Vector2(static_cast<float>(sampleId[0]),
                                     static_cast<float>(sampleId[1])) * delta;
    sampleDistance += (randomOffset * delta);
    Vector3 samplePoint = bottomLeft + ((sampleDistance[0] * right) +
                                        (sampleDistance[1] * up));
    Vector3 rayDir = (samplePoint - position).Normalize();

    // Initialize Ray
    ray.ray = RayF(rayDir, position);
    ray.tMin = nearFar[0];
    ray.tMax = nearFar[1];
}

__device__
inline PrimitiveId GPUCameraPinhole::PrimitiveIndex() const
{
    return 0;
}

inline CPUCameraGroupPinhole::CPUCameraGroupPinhole()    
    : dGPUCameras(nullptr)
    , cameraCount(0)
{}

inline const char* CPUCameraGroupPinhole::Type() const
{
    return TypeName();
}

inline const GPUCameraList& CPUCameraGroupPinhole::GPUCameras() const
{
    return gpuCameraList;
}

inline uint32_t CPUCameraGroupPinhole::CameraCount() const
{
    return cameraCount;
}

inline size_t CPUCameraGroupPinhole::UsedGPUMemory() const
{
    return memory.Size();
}

inline size_t CPUCameraGroupPinhole::UsedCPUMemory() const
{  
    return (sizeof(HitKey) * hHitKeys.size() +
            sizeof(uint16_t) * hMediumIds.size() +
            sizeof(TransformId) * hTransformIds.size() +
            sizeof(Data) * hCameraData.size());
}

static_assert(IsTracerClass<CPUCameraGroupPinhole>::value,
              "CPUCameraPinhole is not a tracer class");