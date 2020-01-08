#pragma once

#include "TracerLib/TracerLogicP.cuh"
#include "TracerLib/TypeTraits.h"

#include "RayAuxStruct.h"

__device__ __host__
inline void RayInitBasic(RayAuxBasic* gOutBasic,
                         const uint32_t writeLoc,
                         // Input
                         const RayAuxBasic& defaults,
                         const RayReg& ray,
                         // Index
                         const uint32_t localPixelId,
                         const uint32_t pixelSampleId)
{
    RayAuxBasic init = defaults;
    init.pixelId = localPixelId;
    init.pixelSampleId = pixelSampleId;

    gOutBasic[writeLoc] = init;
}

class TracerBasic final : public TracerBaseLogic<RayAuxBasic, RayInitBasic>
{
    public:
        static constexpr const char* TypeName() { return "Test"; }

    private:
        static constexpr RayAuxBasic    initals = {Vector3f(1.0f, 1.0f, 1.0f), 0, 0, 1, false};

    protected:
    public:
        // Constructors & Destructor
                        TracerBasic(GPUBaseAcceleratorI& ba,
                                    AcceleratorGroupList&& ag,
                                    AcceleratorBatchMappings&& ab,
                                    MaterialGroupList&& mg,
                                    MaterialBatchMappings&& mb,
                                    GPUEventEstimatorI& ee,
                                    //
                                    const TracerParameters& parameters,
                                    uint32_t hitStructSize,
                                    const Vector2i maxMats,
                                    const Vector2i maxAccels,
                                    const HitKey baseBoundMatKey);
                        ~TracerBasic() = default;

        TracerError     Initialize() override;

        uint32_t        GenerateRays(const CudaSystem& cudaSystem, 
                                     //
                                     RayMemory&, RNGMemory&,
                                     const GPUSceneI& scene,
                                     const CameraPerspective&,
                                     int samplePerLocation,
                                     Vector2i resolution,
                                     Vector2i pixelStart = Zero2i,
                                     Vector2i pixelEnd = BaseConstants::IMAGE_MAX_SIZE) override;
};

static_assert(IsTracerClass<TracerBasic>::value,
              "TracerBasic is not a Tracer Class.");