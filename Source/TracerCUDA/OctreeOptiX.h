#pragma once

#include "OptixSystem.h"
#include "OctreeOptiXPTX.cuh"
#include "AnisoSVO.cuh"
#include "ImageStructs.h"
#include "WFPGCommon.h"

// TODO: Change the include chain make appropriate types common
#include "RayCasterOptiX.h"

class GPUCameraI;

class SVOOptixConeCaster
{
    public:
    static constexpr const char* MODULE_BASE_NAME       = "OptiXShaders/OctreeOptiXPTX.optixir";
    static constexpr const char* RAYGEN_CAM_FUNC_NAME   = "__raygen__SVOCamTrace";
    static constexpr const char* RAYGEN_RAD_FUNC_NAME   = "__raygen__SVORadGen";
    static constexpr const char* MISS_FUNC_NAME         = "__miss__SVO";
    static constexpr const char* CHIT_FUNC_NAME         = "__closesthit__SVO";
    static constexpr const char* INTERSECT32_FUNC_NAME  = "__intersection__SVOMorton32";
    static constexpr const char* INTERSECT64_FUNC_NAME  = "__intersection__SVOMorton64";

    // Implementation Dependant Program Group Indices
    static constexpr uint32_t LEVEL_32_BIT = 0;
    static constexpr uint32_t LEVEL_64_BIT = 1;

    static constexpr int RAD_RAYGEN_PG_INDEX = 0;
    static constexpr int CAM_RAYGEN_PG_INDEX = 1;
    static constexpr int MISS_PG_INDEX = 2;
    static constexpr int MORTON32_HIT_PG_INDEX = 3;
    static constexpr int MORTON64_HIT_PG_INDEX = 4;

    private:
    const OptiXSystem&              optixSystem;

    OptixPipeline                   pipeline;
    OptixModule                     mdl;
    std::vector<OptixProgramGroup>  programGroups;
    // Host copy of the Params
    OctreeAccelParams               hSVOOptixLaunchParams;
    // Device copy of the params
    OctreeAccelParams*              dSVOOptixLaunchParams;
    DeviceLocalMemory               paramsMemory;
    // SBT
    OptixShaderBindingTable         sbtRadGen;
    OptixShaderBindingTable         sbtCamGen;
    DeviceLocalMemory               sbtMemory;
    DeviceLocalMemory               mortonMemory;
    // TODO: Change this to more dynamic representation
    // for Multi-GPU Systems
    std::vector<size_t>                     mortonByteOffsets;
    std::vector<OptixTraversableHandle>     svoLevelAccelerators;
    std::vector<DeviceLocalMemory>          svoLevelAcceleratorMemory;

    protected:
    public:
    // Constructors & Destructor
                            SVOOptixConeCaster(const OptiXSystem&);
                            SVOOptixConeCaster(const SVOOptixConeCaster&) = delete;
                            SVOOptixConeCaster(SVOOptixConeCaster&&) = delete;
    SVOOptixConeCaster&     operator=(const SVOOptixConeCaster&) = delete;
    SVOOptixConeCaster&     operator=(SVOOptixConeCaster&&) = delete;
                            ~SVOOptixConeCaster();

    // Generate the Traversable
    void                    GenerateSVOTraversable(const AnisoSVOctreeCPU&);

    // Calls
    // Cone Trace call for debugging
    void                    ConeTraceFromCamera(// Output
                                                CamSampleGMem<Vector3f> gSamples,
                                                // Input
                                                const GPUCameraI* gCamera,
                                                WFPGRenderMode mode,
                                                uint32_t maxQueryLevelOffset,
                                                const Vector2i& totalPixelCount);


    // Generate radiance map over the spatial locations
    // on the scene
    //void RadianceMapGen(uint32_t* dPartitionOffsets,
    //                    uint32_t* dPartitionBinIds,

    //                    // Range over this id/offsets
    //                    uint32_t segmentStart,
    //                    uint32_t segmentEnd,

    //                    Vector2i radianceMapSize,
    //                    );
};