#pragma once

#include "RayTracer.h"
#include "WorkPool.h"

class DirectTracer : public RayTracer
{
    public:
        static constexpr const char*    TypeName() { return "DirectTracer"; }
        static constexpr const char*    RENDER_TYPE_NAME = "RenderType";

        enum RenderType
        {
            RENDER_FURNACE,
            RENDER_POSITION,
            RENDER_NORMAL,
            RENDER_LIN_DEPTH,
            RENDER_LOG_DEPTH,

            END
        };       

        struct Options
        {
            int32_t     sampleCount = 1;    // Per-axis sample per pixel
            RenderType  renderType = RenderType::RENDER_FURNACE;
        };

    private:
        static const std::array<std::string, RenderType::END> RenderTypeNames;

        Options                 options;
        WorkBatchMap            workMap;
        WorkPool<>              workPool;

        static TracerError      StringToRenderType(RenderType&, const std::string&);
        static std::string      RenderTypeToString(RenderType);

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