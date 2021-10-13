#pragma once

#include "RayTracer.h"
#include "WorkPool.h"

#include "EmptyMaterial.cuh"
#include "GPUPrimitiveEmpty.h"

class DirectTracer : public RayTracer
{
    public:
        static constexpr const char*    TypeName() { return "DirectTracer"; }
        static constexpr const char*    RENDER_TYPE_NAME = "RenderType";

        enum RenderType
        {
            RENDER_FURNACE,
            RENDER_POSITION,
            RENDER_WORLD_NORMAL,
            RENDER_LIN_DEPTH,
            RENDER_LOG_DEPTH,

            END
        };

        struct Options
        {
            int32_t     sampleCount     = 1;                            // Per-axis sample per pixel
            RenderType  renderType      = RenderType::RENDER_FURNACE;   // What to render?
        };

        static TracerError      StringToRenderType(RenderType&, const std::string&);
        static std::string      RenderTypeToString(RenderType);

    private:
        static const std::array<std::string, RenderType::END> RenderTypeNames;

        Options                 options;
        WorkBatchMap            workMap;

        // Work Pools
        WorkPool<>              furnaceWorkPool;
        WorkPool<>              normalWorkPool;
        WorkPool<>              positionWorkPool;

        // Empty Prim and Material (For generating custom work for hits)
        GPUPrimitiveEmpty       emptyPrim;
        EmptyMat<BasicSurface>  emptyMat;

        uint32_t                currentCameraId;

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
        void                    GenerateWork(const VisorTransform&, int cameraId) override;
        void                    GenerateWork(const GPUCameraI&) override;
        bool                    Render() override;
};

static_assert(IsTracerClass<DirectTracer>::value,
              "DirectTracer is not a Tracer Class.");