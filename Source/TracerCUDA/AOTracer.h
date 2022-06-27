#pragma once
#include "RayTracer.h"
#include "WorkPool.h"
#include "GPULightI.h"

#include "EmptyMaterial.cuh"
#include "GPUPrimitiveEmpty.h"

class AOTracer final : public RayTracer
{
    public:
        static constexpr const char*    TypeName() { return "AOTracer"; }

        static constexpr const char*    MAX_DISTANCE_NAME = "MaxDistance";

        struct Options
        {
            int32_t     sampleCount = 1;
            float       maxDistance = 2.0f;
        };

    private:
        Options                 options;
        WorkBatchMap            workMap;

        // AOMiss HitKey
        HitKey                  aoMissKey;

        // Work pool
        WorkPool<>              workPool;

        // Empty Prim and Material (For generating custom work for hits)
        GPUPrimitiveEmpty       emptyPrim;
        EmptyMat<BasicSurface>  emptyMat;

        // States
        bool                    hitPhase;
        uint32_t                depth;

    protected:
    public:
        // Constructors & Destructor
                                AOTracer(const CudaSystem&,
                                         const GPUSceneI&,
                                         const TracerParameters&);
                                ~AOTracer() = default;

        TracerError             Initialize() override;
        TracerError             SetOptions(const OptionsI&) override;
        void                    AskOptions() override;

        void                    GenerateWork(uint32_t cameraIndex) override;
        void                    GenerateWork(const VisorTransform&, uint32_t cameraIndex) override;
        void                    GenerateWork(const GPUCameraI&) override;
        bool                    Render() override;
        void                    Finalize() override;
};

static_assert(IsTracerClass<AOTracer>::value,
              "AOTracer is not a Tracer Class.");