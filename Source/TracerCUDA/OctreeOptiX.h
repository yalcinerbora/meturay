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
    static constexpr const char* RAYGEN_CAM_FUNC_NAME   = "__raygen__SVOCam";
    static constexpr const char* RAYGEN_RAD_FUNC_NAME   = "__raygen__SVORadiance";
    static constexpr const char* MISS_FUNC_NAME         = "__miss__SVO";
    static constexpr const char* CHIT_FUNC_NAME         = "__closesthit__SVO";
    static constexpr const char* INTERSECT_FUNC_NAME    = "__intersection__SVO";

    static constexpr int CH_INDEX = 0;
    static constexpr int AH_INDEX = 1;
    static constexpr int INTS_INDEX = 2;

    private:
    const OptiXSystem&          optixSystem;


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
    // TODO: Change this to more dynamic representation
    // for Multi-GPU Systems
    std::vector<OptixTraversableHandle>     svoLevelAccelerators;

    protected:
    public:
    // Constructors & Destructor
    SVOOptixConeCaster(const OptiXSystem&);

    // Generate the Traversable
    void GenerateSVOTraversable(const AnisoSVOctreeCPU&);

    // Calls
    // Cone Trace call for debugging
    void ConeTraceFromCamera(// Output
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