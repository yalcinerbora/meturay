#pragma once
#include "RayTracer.h"
#include "WorkPool.h"
#include "GPULightI.h"

class AOTracer final : public RayTracer
{
    public:
        static constexpr const char*    TypeName() { return "PathTracer"; }

        static constexpr const char*    MAX_DISTANCE_NAME = "MaxDistance";

        struct Options
        {
            int32_t     sampleCount = 1;
            float       maxDistance = 2.0f;
        };

    private:
        Options                 options;
        WorkBatchMap            workMap;

        // Generic work pool
        WorkPool<>              workPool;
        // Light material work pool
        WorkPool<>              lightWorkPool;
        
    protected:
    public:
        // Constructors & Destructor
                                AOTracer(const CudaSystem&,
                                           const GPUSceneI&,
                                           const TracerParameters&);
                                ~AOTracer() = default;

        TracerError             Initialize() override;
        TracerError             SetOptions(const TracerOptionsI&) override;
        void                    AskOptions() override;

        void                    GenerateWork(int cameraId) override;
        void                    GenerateWork(const VisorCamera&) override;
        bool                    Render() override;
};

static_assert(IsTracerClass<AOTracer>::value,
              "AOTracer is not a Tracer Class.");