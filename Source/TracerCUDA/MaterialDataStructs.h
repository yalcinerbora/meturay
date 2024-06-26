#pragma once

#include "RayLib/Vector.h"
#include "GPUMediumI.h"
#include "TextureReference.cuh"

struct NullData {};

struct AlbedoMatData
{
    const Vector3* dAlbedo;
};

struct ReflectMatData
{
    const Vector4* dAlbedoAndRoughness;
};

struct RefractMatData
{
    const Vector3* dAlbedo;
    const uint32_t* mediumIndices;

    // Global Medium Array pointer
    const GPUMediumI* const* dMediums;
    uint32_t baseMediumIndex;
};

struct UnrealMatData
{
    const TextureRefI<2, Vector3f>** dNormal;
    const TextureRefI<2, Vector3f>** dAlbedo;
    const TextureRefI<2, float>**    dRoughness;
    const TextureRefI<2, float>**    dMetallic;
    const TextureRefI<2, float>**    dSpecular;
};

struct LambertMatData
{
    const TextureRefI<2, Vector3f>** dAlbedo;
    const TextureRefI<2, Vector3f>** dNormal;
};

struct MetalMatData
{
    // TODO: Change to surface varying data
    const Vector3f*     dEta;
    const Vector3f*     dK;
    const Vector3f*     dSpecular;
    const float*        dRoughness;

    //const TextureRefI<2, Vector3f>**    dNormal;

    //const TextureRefI<2, Vector3f>**    dEta;
    //const TextureRefI<2, Vector3f>**    dK;
    //const TextureRefI<2, float>**       dRoughness;
};

//struct GlassMatData
//{
//    const TextureRefI<2, Vector3f>** dEta;
//    const TextureRefI<2, Vector3f>** dK;
//    const TextureRefI<2, float>** dRoughness;
//
//    const uint32_t* mediumIndices;
//
//    // Global Medium Array pointer
//    const GPUMediumI* const* dMediums;
//    uint32_t baseMediumIndex;
//};

using EmissiveMatData = AlbedoMatData;