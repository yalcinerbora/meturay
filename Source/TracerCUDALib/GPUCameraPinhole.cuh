#pragma once

#include "GPUCameraI.h"
#include "DeviceMemory.h"
#include "GPUTransformI.h"

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
                                             const float& aperture,
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

        static constexpr const char* NAME_APERTURE  = "apertureSize";
        static constexpr const char* NAME_FOCUS     = "focusDistance";

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
            float                   aperture;
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