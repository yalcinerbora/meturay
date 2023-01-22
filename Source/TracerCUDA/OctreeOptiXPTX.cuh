#pragma once

#include <optix_device.h>
#include "AnisoSVO.cuh"

struct OctreeAccelParams
{
    const OptixTraversableHandle*   octreeLevelBVHs;
    // Put everything to here
    // This holds many information
    AnisoSVOctreeGPU                svo;

    // Output buffers
};

// SVO Hit Record
// We only require the leaf id so nothing is held here

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SVOEmptyRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // Optix 7 Course had dummy pointer here so i will leave it as well
    void* empty;
};

// ExternCWrapper Macro
#define WRAP_FUCTION(NAME, FUNCTION) \
    extern "C" __global__ void NAME(){FUNCTION();}