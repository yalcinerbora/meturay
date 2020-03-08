#pragma once

#include "TracerLib/GPUTracer.h"
#include "TracerLib/TypeTraits.h"

#include "RayAuxStruct.h"
 
class TestPathTracer final : public GPUTracer

{
    public:
        static constexpr const char*    TypeName() { return "Test"; }

    private:
        DeviceMemory    auxBuffer0;
        DeviceMemory    auxBuffer1;

        RayAuxBasic*    auxIn;
        RayAuxBasic*    auxOut;        


    protected:
    public:
        // Constructors & Destructor
                        TestPathTracer();
                        ~TestPathTracer() = default;

        //TracerError     Initialize(GPUScene&) override;

        //uint32_t        GenerateWork(const CudaSystem& cudaSystem, 
        //                             //
        //                             
        //                              RNGMemory&,
        //                             const GPUSceneI& scene,
        //                             const CameraPerspective&,
        //                             int samplePerLocation,
        //                             Vector2i resolution,
        //                             Vector2i pixelStart = Zero2i,
        //                             Vector2i pixelEnd = BaseConstants::IMAGE_MAX_SIZE) override;
};

//static_assert(IsTracerClass<TestPathTracer>::value,
//              "TracerBasic is not a Tracer Class.");