#pragma once
/**

Tracer Partial Implementation for CUDA arcitechture

It utilizes hitting, ray memory and image memory allocation.

This is multi-gpu aware implementation.

All Tracers should inherit this class

*/

#include <deque>
#include <functional>
#include <set>
#include <map>
#include <sstream>

#include "RayLib/TracerStructs.h"
#include "RayLib/GPUTracerI.h"
#include "RayLib/VisorTransform.h"

#include "RNGMemory.h"
#include "RayMemory.h"
#include "ImageMemory.h"

class GPUMediumI;
class GPUTransformI;
class GPUSceneI;
class CudaSystem;

class GPUEndpointI;
class GPUCameraI;
class GPULightI;

struct TracerError;

class GPUTracer : public GPUTracerI
{
    private:
        static constexpr const size_t   AlignByteCount = 128;

        //
        const uint32_t                              maxHitSize;
        // Batches of Accelerator
        GPUBaseAcceleratorI&                        baseAccelerator;
        const AcceleratorBatchMap&                  accelBatches;
        // Batches of Material
        const std::map<NameGPUPair, GPUMatGPtr>&    materialGroups;
        const NamedList<CPUTransformGPtr>&          transforms;
        const NamedList<CPUMediumGPtr>&             mediums;
        const NamedList<CPULightGPtr>&              lights;
        const NamedList<CPUCameraGPtr>&             cameras;
        const WorkBatchCreationInfo&                workInfo;
        // Current camera index for copying the camera for transform
        uint32_t                                    currentCameraIndex;

        // GPU Memory
        DeviceMemory                                tempTransformedCam;
        DeviceMemory                                commonTypeMemory;

        TracerError                                 LoadCameras(std::vector<const GPUCameraI*>&,
                                                                std::vector<const GPUEndpointI*>&);
        TracerError                                 LoadLights(std::vector<const GPULightI*>&,
                                                               std::vector<const GPUEndpointI*>&);
        TracerError                                 LoadTransforms(std::vector<const GPUTransformI*>&);
        TracerError                                 LoadMediums(std::vector<const GPUMediumI*>&);

    protected:
        // Max Bit Sizes for Efficient Sorting
        Vector2i                            maxAccelBits;
        Vector2i                            maxWorkBits;
        // Cuda System For Kernel Calls
        const CudaSystem&                   cudaSystem;
        // GPU Memory
        RNGMemory                           rngMemory;
        RayMemory                           rayMemory;
        ImageMemory                         imgMemory;
        const GPUTransformI**               dTransforms;
        const GPUMediumI**                  dMediums;
        const GPUCameraI**                  dCameras;
        const GPULightI**                   dLights;
        const GPUEndpointI**                dEndpoints;
        // Camera Related Helper Types for fast access some data
        std::vector<VisorTransform>         cameraVisorTransforms;
        std::vector<std::string>            cameraGroupNames;
        // Indices for Identity Transform &
        // Base Medium
        const uint32_t                      baseMediumIndex;
        const uint32_t                      identityTransformIndex;
        const uint32_t                      boundaryTransformIndex;
        //
        TracerParameters                    params;
        //
        uint32_t                            currentRayCount;
        uint32_t                            transformCount;
        uint32_t                            mediumCount;
        uint32_t                            lightCount;
        uint32_t                            cameraCount;
        uint32_t                            endpointCount;
        // Callbacks
        TracerCallbacksI*                   callbacks;
        bool                                crashed;
        // Current Work Partition
        RayPartitions<uint32_t>             workPartition;

        // Interface
        virtual void                        ResetHitMemory(uint32_t rayCount,
                                                           HitKey baseBoundMatKey);

        // Do a hit determination over current rays
        void                                HitAndPartitionRays();
        // Determine auxiliary size
        void                                WorkRays(const WorkBatchMap&,
                                                     const RayPartitionsMulti<uint32_t>& outPortions,
                                                     uint32_t totalRayOut,
                                                     HitKey baseBoundMatKey);
        // Convert GPUCamera to VisorCam
        VisorTransform                      SceneCamTransform(uint32_t cameraIndex);
        const GPUCameraI*                   GenerateCameraWithTransform(const VisorTransform&,
                                                                        uint32_t cameraIndex);

        // Internals
        template <class... Args>
        void                                SendLog(const char*, Args...);
        void                                SendError(TracerError e, bool isFatal);

        RayPartitionsMulti<uint32_t>        PartitionOutputRays(uint32_t& totalOutRay,
                                                                const WorkBatchMap&) const;

        static Vector2i                     DetermineMaxBitFromId(const Vector2i&);

    public:
        // Constructors & Destructor
                                            GPUTracer(const CudaSystem&,
                                                      const GPUSceneI&,
                                                      const TracerParameters&);
                                            GPUTracer(const GPUTracer&) = delete;
                                            GPUTracer(GPUTracer&&) = delete;
        GPUTracer&                          operator=(const GPUTracer&) = delete;
        GPUTracer&                          operator=(GPUTracer&&) = delete;
                                            ~GPUTracer() = default;

        // =====================//
        // RESPONSE FROM TRACER //
        // =====================//
        // Callbacks
        void                    AttachTracerCallbacks(TracerCallbacksI&) override;

        // ===================//
        // COMMANDS TO TRACER //
        // ===================//
        TracerError             Initialize() override;
        void                    SetParameters(const TracerParameters&) override;
        void                    AskParameters() override;
        void                    Finalize() override;

        // Image Related
        void                    SetImagePixelFormat(PixelFormat) override;
        void                    ReportionImage(Vector2i start = Zero2i,
                                               Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        void                    ResizeImage(Vector2i resolution) override;
        void                    ResetImage() override;
};

inline void GPUTracer::AttachTracerCallbacks(TracerCallbacksI& tc)
{
    callbacks = &tc;
}