#pragma once

#include <optix_device.h>
#include "AnisoSVO.cuh"
#include "ImageStructs.h"
#include "RayAuxStruct.cuh"
#include "WFPGCommon.h"
#include "SVOOptiXRadianceBuffer.cuh"

struct OctreeAccelParams
{
    template<class T>
    using SegmentedField = SVOOptixRadianceBuffer::SegmentedField<T>;

    //===============//
    //     Common    //
    //===============//
    const OptixTraversableHandle*   octreeLevelBVHs;
    OptixTraversableHandle          sceneBVH;
    bool                            utilizeSceneAccelerator;
    // Put everything to here
    // This holds many information
    AnisoSVOctreeGPU                svo;
    // Aperture (in solid angles) of the pixel (when debugging)
    // or cone when generating radiance fields
    float                           pixelOrConeAperture;

    union
    {
        //===================//
        // Cam Trace Related //
        //===================//
        struct
        {
            const RayGMem*          gRays;          // Generated rays from separate kernel
                                                    // (Not generating here because of virtual functions)
            const RayAuxWFPG*       gRayAux;        // Ray auxiliary data (only using sample index)
            CamSampleGMem<Vector4f> gSamples;       // Sample Buffer (for writing)
            // Constants
            WFPGRenderMode          renderMode;     // What to output? (False_color, normal, radiance, etc)
            uint32_t                maxQueryOffset; // Do not query below this level
        } ct;
        //======================//
        // Radiance Gen Related //
        //======================//
        struct
        {
            SegmentedField<float*>  fieldSegments;
            // All of these are determined per bin
            const Vector4f*         dRadianceFieldRayOrigins;
            const Vector2f*         dProjJitters;
            int32_t                 binOffset;
        } rg;
    };
    //....
};

// SVO Hit Record
// We only require the leaf id so nothing is held here
template <class T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SVOHitRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

    T* dMortonCode;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SVOEmptyRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // Optix 7 Course had dummy pointer here so i will leave it as well
    void* empty;
};

// ExternCWrapper Macro
#define WRAP_FUCTION(NAME, FUNCTION) \
    extern "C" __global__ void NAME(){FUNCTION();}