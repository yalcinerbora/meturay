#pragma once

#include "TracerLib/TracerLogicP.cuh"
#include "TracerLib/TypeTraits.h"

#include "RayAuxStruct.h"

class TracerBasic final : 
    public TracerBaseLogic<RayAuxBasic>,
    public 
{
    public:
        static constexpr const char*    TypeName() { return "Test"; }

    private:
        ImageMemory     img;
        GPURayTracer    rayTracer;

    protected:
    public:
        // Constructors & Destructor
                        TracerBasic(.............,
                                    GPUBaseAcceleratorI& ba,
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

        uint32_t        GenerateWork(const CudaSystem& cudaSystem, 
                                     //
                                     
                                      RNGMemory&,
                                     const GPUSceneI& scene,
                                     const CameraPerspective&,
                                     int samplePerLocation,
                                     Vector2i resolution,
                                     Vector2i pixelStart = Zero2i,
                                     Vector2i pixelEnd = BaseConstants::IMAGE_MAX_SIZE) override;
};

static_assert(IsTracerClass<TracerBasic>::value,
              "TracerBasic is not a Tracer Class.");