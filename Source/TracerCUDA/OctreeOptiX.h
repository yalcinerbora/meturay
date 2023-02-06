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
    static constexpr const char* CHIT_SCENE_FUNC_NAME   = "__closesthit__Scene";
    static constexpr const char* INTERSECT32_FUNC_NAME  = "__intersection__SVOMorton32";
    static constexpr const char* INTERSECT64_FUNC_NAME  = "__intersection__SVOMorton64";

    static constexpr const char* PARAMS_BUFFER          = "params";

    // Implementation Dependant Program Group Indices
    static constexpr uint32_t LEVEL_32_BIT = 0;
    static constexpr uint32_t LEVEL_64_BIT = 1;

    static constexpr int RAD_RAYGEN_PG_INDEX = 0;
    static constexpr int CAM_RAYGEN_PG_INDEX = 1;
    static constexpr int MISS_PG_INDEX = 2;
    static constexpr int MORTON32_HIT_PG_INDEX = 3;
    static constexpr int MORTON64_HIT_PG_INDEX = 4;
    static constexpr int SCENE_HIT_PG_INDEX = 5;

    private:
    // SVO
    const AnisoSVOctreeCPU&         svoCPU;
    // OptiX Related
    const OptiXSystem&              optixSystem;

    OptixPipeline                   pipeline;
    OptixModule                     mdl;
    std::vector<OptixProgramGroup>  programGroups;
    // Device copy of the params
    OctreeAccelParams               hOptixLaunchParams;
    DeviceLocalMemory               paramsMemory;
    OctreeAccelParams*              dOptixLaunchParams;
    OptixTraversableHandle*         dOptixTraversables;
    // SBT
    OptixShaderBindingTable         sbtRadGen;
    OptixShaderBindingTable         sbtRadGenScene;
    OptixShaderBindingTable         sbtCamGen;
    OptixShaderBindingTable         sbtCamGenScene;
    DeviceLocalMemory               sbtMemory;
    // Morton Codes on the records
    DeviceMemory                    mortonMemory;
    std::vector<size_t>             mortonByteOffsets;
    // TODO: Change this to more dynamic representation
    // for Multi-GPU Systems
    std::vector<OptixTraversableHandle>     svoLevelAccelerators;
    std::vector<DeviceLocalMemory>          svoLevelAcceleratorMemory;

    // DEBUG
    // Entire Scene Traversable
    OptixTraversableHandle                  sceneTraversable;
    uint32_t                                sceneSBTCount;

    protected:
    public:
    // Constructors & Destructor
                            SVOOptixConeCaster(const OptiXSystem&,
                                               const GPUBaseAcceleratorI& baseAccelerator,
                                               const AnisoSVOctreeCPU&);
                            SVOOptixConeCaster(const SVOOptixConeCaster&) = delete;
                            SVOOptixConeCaster(SVOOptixConeCaster&&) = delete;
    SVOOptixConeCaster&     operator=(const SVOOptixConeCaster&) = delete;
    SVOOptixConeCaster&     operator=(SVOOptixConeCaster&&) = delete;
                            ~SVOOptixConeCaster();

    // Generate the Traversable
    void                    GenerateSVOTraversable();

    // Calls
    // Cone Trace call for debugging
    void                    ConeTraceFromCamera(// Output
                                                CamSampleGMem<Vector4f> gSamples,
                                                // Input
                                                const RayGMem* gRays,
                                                const RayAuxWFPG* gRayAux,
                                                WFPGRenderMode mode,
                                                uint32_t maxQueryLevelOffset,
                                                bool useSceneAccelerator,
                                                float pixelAperture,
                                                const uint32_t totalRayCount);


    // Generate radiance map over the spatial locations
    // on the scene
    void                    CopyRadianceMapGenParams(const Vector4f* dRadianceFieldRayOrigins,
                                                     const float* dProjJitters,
                                                     SVOOptixRadianceBuffer::SegmentedField<float*>,
                                                     bool useSceneAccelerator,
                                                     float coneAperture);
    void                    RadianceMapGen(uint32_t segmentOffset,
                                           uint32_t totalRayCount);
};