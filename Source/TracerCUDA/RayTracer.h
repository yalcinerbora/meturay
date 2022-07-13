#pragma once

#include "GPUTracer.h"
#include "TypeTraits.h"
#include "GPUCameraI.h"
#include "CameraFunctions.h"
#include "RNGIndependent.cuh"

// Generic Ray Tracer Class
class RayTracer : public GPUTracer
{
    public:
        // Option Names
        static constexpr const char* SAMPLE_NAME = "Samples";

    private:
        void                            AllocateSampleBuffer(uint32_t totalSampleCount);

    protected:
        // Auxiliary Data for Each Ray
        DeviceMemory                    auxBuffer0;
        DeviceMemory                    auxBuffer1;
        //
        DeviceMemory*                   dAuxIn;
        DeviceMemory*                   dAuxOut;
        // Sample Buffer of each ray
        DeviceMemory                    dSampleBuffer;
        CamSampleGMem<Vector4f>         dSamplePtrs;
        CamSampleGMemConst<Vector4f>    dSampleConstPtrs;

        // Scene Reference
        const GPUSceneI&        scene;

        int32_t                 totalSamplePerPixel;

        template <class AuxStruct, class AuxInitFunctor, class RNG>
        void                    GenerateRays(uint32_t sceneCamId,
                                             int32_t sampleCount,
                                             const AuxInitFunctor& initFunctor,
                                             bool incSampleCount = true,
                                             bool antiAliasOn = true);
        template <class AuxStruct, class AuxInitFunctor, class RNG>
        void                    GenerateRays(const GPUCameraI& dCamera,
                                             int32_t sampleCount,
                                             const AuxInitFunctor& initFunctor,
                                             bool incSampleCount = true,
                                             bool antiAliasOn = true);
        template <class AuxStruct, class AuxInitFunctor, class RNG>
        void                    GenerateRays(const VisorTransform& transform, uint32_t sceneCamId,
                                             int32_t sampleCount,
                                             const AuxInitFunctor& initFunctor,
                                             bool incSampleCount = true,
                                             bool antiAliasOn = true);
        void                    SwapAuxBuffers();

        void                    UpdateFrameAnalytics(const std::string& throughputSuffix,
                                                     uint32_t spp);

    public:
        // Constructors & Destructor
                                RayTracer(const CudaSystem&,
                                          const GPUSceneI&,
                                          const TracerParameters&);
                                ~RayTracer() = default;

        TracerError             Initialize() override;

        size_t                  TotalGPUMemoryUsed() const override;
};