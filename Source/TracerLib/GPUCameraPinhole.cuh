#pragma once

#include "GPUCameraI.cuh"
#include "DeviceMemory.h"

class GPUCameraPinhole : public GPUCameraI
{
    public:
        struct Data
        {
            // Sample Ready Parameters
            // All of which is world space
            Vector3     position;
            Vector3     right;
            Vector3     up;
            Vector3     bottomLeft;         // Far plane bottom left
            Vector2     planeSize;          // Far plane size
            Vector2     nearFar;
        };

    private:
        const Data&             data;
        const GPUTransformI&    transform;

    protected:
    public:
        // Constructors & Destructor
        __device__          GPUCameraPinhole(const Data&,
                                             uint16_t mediumId,
                                             HitKey materialKey,
                                             const GPUTransformI& transform);
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

class CPUCameraPinhole : public CPUCameraGroupI
{
    public:
        static const char* TypeName() { return "Single"; }
        // Node Names
        static constexpr const char* APERTURE_NAME  = "apertureSize";
        static constexpr const char* FOCUS_NAME     = "focusDistance";
        static constexpr const char* PLANES_NAME    = "planes";
        static constexpr const char* FOV_NAME       = "fov";
        static constexpr const char* GAZE_NAME      = "gaze";
        static constexpr const char* UP_NAME        = "up";

    private:
        DeviceMemory                    memory;
        const GPUCameraPinhole::Data*   dCamData;
        const GPUCameraPinhole*         dGPUCameras;
        GPUCameraList                   gpuCameraList;
        uint32_t                        cameraCount;

        // Global Transform Array
        const GPUTransformI**           dGPUTransforms;

    protected:
    public:
        // Constructors & Destructor
                                        CPUCameraPinhole();
                                        ~CPUCameraPinhole() = default;

        // Interface
        const char*                     Type() const override;
        const GPUCameraList&            GPUCameras() const override;
        SceneError					    InitializeGroup(const NodeListing& lightNodes,
                                                        const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                        const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                        const MaterialKeyListing& allMaterialKeys,
                                                        double time,
                                                        const std::string& scenePath) override;
        SceneError					    ChangeTime(const NodeListing& lightNodes, double time,
                                                   const std::string& scenePath) override;
        TracerError					    ConstructCameras(const CudaSystem&) override;
        uint32_t					    CameraCount() const override;

        size_t						    UsedGPUMemory() const override;
        size_t						    UsedCPUMemory() const override;

        void						    AttachGlobalTransformArray(const GPUTransformI** deviceTranfsorms) override;
};

__device__
inline GPUCameraPinhole::GPUCameraPinhole(const Data& d,
                                          uint16_t mediumId,
                                          HitKey materialKey,
                                          const GPUTransformI& transform)
    : GPUEndpointI(materialKey, mediumId)
    , data(d)
    , transform(transform)
{

    //CPUCamera;

    //// Find world space window sizes
    //float widthHalf = tanf(data.fov[0] * 0.5f) * data.nearPlane;
    //float heightHalf = tanf(data.fov[1] * 0.5f) * data.nearPlane;

    //// Camera Vector Correction
    //Vector3 gaze = data.gazePoint - data.position;
    //right = Cross(gaze, cam.up).Normalize();
    //up = Cross(right, gaze).Normalize();
    //gaze = Cross(up, right).Normalize();

    //// Camera parameters
    //bottomLeft = (cam.position
    //              - right * widthHalf
    //              - up * heightHalf
    //              + gaze * cam.nearPlane);
    //position = cam.position;
    //planeSize = Vector2(widthHalf, heightHalf) * 2.0f;
    //nearFar = Vector2(cam.nearPlane, cam.farPlane);
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
    direction = sampleLoc - data.position;
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
    Vector2 delta = Vector2(data.planeSize[0] / static_cast<float>(sampleMax[0]),
                            data.planeSize[1] / static_cast<float>(sampleMax[1]));

    // Create random location over sample rectangle
    float dX = GPUDistribution::Uniform<float>(rng);
    float dY = GPUDistribution::Uniform<float>(rng);
    Vector2 randomOffset = Vector2(dX, dY);
    //Vector2 randomOffset = Vector2(0.5f);

    Vector2 sampleDistance = Vector2(static_cast<float>(sampleId[0]),
                                     static_cast<float>(sampleId[1])) * delta;
    sampleDistance += (randomOffset * delta);
    Vector3 samplePoint = data.bottomLeft + (sampleDistance[0] * data.right)
                                     + (sampleDistance[1] * data.up);
    Vector3 rayDir = (samplePoint - data.position).Normalize();

    // Initialize Ray
    ray.ray = RayF(rayDir, data.position);
    ray.tMin = data.nearFar[0];
    ray.tMax = data.nearFar[1];
}

__device__ 
inline PrimitiveId GPUCameraPinhole::PrimitiveIndex() const
{
    return 0;
}

inline CPUCameraPinhole::CPUCameraPinhole()
    : dCamData(nullptr)
    , dGPUCameras(nullptr)
    , dGPUTransforms(nullptr)
{}

inline const char* CPUCameraPinhole::Type() const
{
    return TypeName();
}

inline const GPUCameraList& CPUCameraPinhole::GPUCameras() const
{
    return gpuCameraList;
}

inline uint32_t CPUCameraPinhole::CameraCount() const
{
    return cameraCount;
}

inline size_t CPUCameraPinhole::UsedGPUMemory() const
{
    return memory.Size();
}

inline size_t CPUCameraPinhole::UsedCPUMemory() const
{
    return  0;
}

inline void CPUCameraPinhole::AttachGlobalTransformArray(const GPUTransformI** deviceTranfsorms)
{
    dGPUTransforms = deviceTranfsorms;
}