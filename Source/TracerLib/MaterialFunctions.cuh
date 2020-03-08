#pragma once

#include "RayLib/Ray.h"
#include "RayLib/HitStructs.h"

class RandomGPU;

// METURay only supports
// 2D Textures 2^24 different textures layers (per texture) and 255 different mips per texture.
// For research purposes this should be enough.
using TextureId = HitKeyT<uint32_t, 8u, 24u>;
struct UVList
{
    TextureId id;   // 24-bit layer id, 8-bit mip id
    Vector2us uv;   // UNorm 2x16 data
};

// Texture System should return another UVList in which
// 24-bit portion returns actual texture Id


// Sample function is responsible for random BRDF evalulation
// and required data access for BRDFs
//
// Returns evaluated BxDF function result, a sampled ray and prob density of the sample.
// it returns a ray since subsurface evaluation makes wo to come out from a different
// location.
// 
// MGroup::Data holds the material information in batched manner
// Function itself is responsible for accessing that structure using matId
//
// This function provided by MaterialGroup.
template <class Data, class Surface>
using SampleFunc = Vector3(*)(// Sampled Output
                              RayF& wo,
                              float& pdf,
                              // Input
                              const Vector3& wi,                // Incoming Radiance
                              const Vector3& pos,               // Position
                              const Surface& surface,
                              const UVList* uvs,
                              // I-O
                              RandomGPU& rng,
                              // Constants
                              const Data&,
                              const HitKey::Type& matId,
                              // For-MultiSample Materials
                              // or materials that uses multiple PDFs
                              // for different segments of the BxDF
                              uint32_t sampleIndex);

// Material provides this in order to for renderer to sample its own strategy
template <class Data, class Surface>
using EvaluateFunc = Vector3(*)(// Input
                                const Vector3& wo,                // Outgoing Radiance
                                const Vector3& wi,                // Incoming Radiance
                                const Vector3& pos,               // Position
                                const Surface& surface, 
                                const UVList* uvs,
                                // Constants
                                const Data&,
                                const HitKey::Type& matId);

//===================================//
// Texture Caching Related Functions //
//===================================//

// This call is per-ray which returns multiple of UVs
// that is dependant of the material.
//
// Each ray returns its own UV location which may be required by the texture cache
// 
template <class Data, class Surface>
using AcquireUVList = void(*)(//Output
                              UVList*, 
                              const Surface& surface,
                              // Constants
                              const Data&,
                              const HitKey::Type& matId);

