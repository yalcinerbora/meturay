#pragma once

#include "RayTracer.h"
#include "TracerLib/WorkPool.h"

class DirectTracer : public RayTracer
{
        public:
        static constexpr const char*    TypeName() { return "TestDirect"; }
        
        struct Options                 
        {
            int32_t     sampleCount = 1;    // Per-axis sample per pixel
        };

    private:
        Options                 options;
        WorkBatchMap            workMap;
        WorkPool<>              workPool;

    protected:
    public:
        // Constructors & Destructor
                                DirectTracer(const CudaSystem&,
                                             const GPUSceneI&,
                                             const TracerParameters&);
                                ~DirectTracer() = default;

        TracerError             Initialize() override;
        TracerError             SetOptions(const TracerOptionsI&) override;
        void                    AskOptions() override;
        //
        void                    GenerateWork(int cameraId) override;
        void                    GenerateWork(const VisorCamera&) override;
        bool                    Render() override;
};

static_assert(IsTracerClass<DirectTracer>::value,
              "DirectTracer is not a Tracer Class.");