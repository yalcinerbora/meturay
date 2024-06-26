#pragma once

#include "GPUTracer.h"
#include "TypeTraits.h"
#include "GPUCameraI.h"
#include "SampleMemory.h"
#include "CameraFunctions.h"
#include "RNGIndependent.cuh"

// Generic Ray Tracer Class
class RayTracer : public GPUTracer
{
    public:
        // Option Names
        static constexpr const char* SAMPLE_NAME = "Samples";

    private:
    protected:
        // Auxiliary Data for Each Ray
        DeviceMemory                    auxBuffer0;
        DeviceMemory                    auxBuffer1;
        //
        DeviceMemory*                   dAuxIn;
        DeviceMemory*                   dAuxOut;
        // Sample Buffer of each ray
        SampleMemory                    sampleMemory;
        GPUReconFilterI*                reconFilter;
        // Scene Reference
        const GPUSceneI&                scene;
        int32_t                         totalSamplePerPixel;

        template <class AuxStruct, class AuxInitFunctor, class RNG, class SampleValType>
        void                    GenerateRays(uint32_t sceneCamId,
                                             int32_t sampleCount,
                                             const AuxInitFunctor& initFunctor,
                                             bool incSampleCount = true,
                                             bool antiAliasOn = true);
        template <class AuxStruct, class AuxInitFunctor, class RNG, class SampleValType>
        void                    GenerateRays(const GPUCameraI& dCamera,
                                             int32_t sampleCount,
                                             const AuxInitFunctor& initFunctor,
                                             bool incSampleCount = true,
                                             bool antiAliasOn = true);
        template <class AuxStruct, class AuxInitFunctor, class RNG, class SampleValType>
        void                    GenerateRays(const VisorTransform& transform, uint32_t sceneCamId,
                                             int32_t sampleCount,
                                             const AuxInitFunctor& initFunctor,
                                             bool incSampleCount = true,
                                             bool antiAliasOn = true);
        void                    SwapAuxBuffers();

        void                    UpdateFrameAnalytics(const std::string& throughputSuffix,
                                                     uint32_t spp);
        void                    ResetSPP();

    public:
        // Constructors & Destructor
                                RayTracer(const CudaSystem&,
                                          const GPUSceneI&,
                                          const TracerParameters&);
                                ~RayTracer() = default;

        TracerError             Initialize() override;

        size_t                  TotalGPUMemoryUsed() const override;

        void                    AttachReconFilter(GPUReconFilterI*) override;

        void                    ResetImage() override;

        // Filtered Finalize
        void                    Finalize() override;


};