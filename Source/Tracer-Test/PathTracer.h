#pragma once

#include "BasicTracer.h"

class PathTracer final : public BasicTracer
{
    public:
        static constexpr const char*    TypeName() { return "Test"; }

        static constexpr const char*    MAX_DEPTH_NAME = "MaxDepth";
        static constexpr const char*    NEE_NAME = "NextEventEstimation";

        struct PTOptions : public Options
        {
            uint32_t                    maximumDepth = 10;
            bool                        nextEventEstimation = true;
        };

    private:
        PTOptions               options;
        uint32_t                currentDepth;

    protected:
    public:
        // Constructors & Destructor
                                PathTracer(CudaSystem&, GPUSceneI&, 
                                            const TracerParameters&);
                                ~PathTracer() = default;

        TracerError             Initialize() override;
        TracerError             SetOptions(const TracerOptionsI&) override;
        void                    AskOptions() override;

        void                    GenerateWork(int cameraId) override;
        void                    GenerateWork(const CPUCamera&) override;
        bool                    Render() override;
};

static_assert(IsTracerClass<PathTracer>::value,
              "TracerBasic is not a Tracer Class.");