#pragma once

#include "RayLib/Ray.h"
#include "GPUMediumI.h"

class RandomGPU;

// IsEmissive function is used to check if this mat has a non zero Le(wo,p) part
template <class Data>
using IsEmissiveFunc = bool(*)(const Data&,
                               const HitKey::Type& matId);

template <class Data, class Surface>
using SpecularityFunc = float(*)(const Surface&,
                                 const Data&,
                                 const HitKey::Type& matId);

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
                              RayF& wo,                         // Out direction
                              float& pdf,                       // PDF for Monte Carlo
                              const GPUMediumI*& outMedium,
                              // Input
                              const Vector3& wi,                // Incoming Radiance
                              const Vector3& pos,               // Position
                              const GPUMediumI& m,
                              //
                              const Surface& surface,           // Surface info (normals uvs etc.)
                              // I-O
                              RandomGPU& rng,
                              // Constants
                              const Data&,
                              const HitKey::Type& matId,
                              // For-MultiSample Materials
                              // or materials that uses multiple PDFs
                              // for different segments of the BxDF
                              uint32_t sampleIndex);

// Direct fetching emission data
template <class Data, class Surface>
using EmissionFunc = Vector3(*)(// Input
                                const Vector3& wo,              // Outgoing Radiance
                                const Vector3& pos,             // Position
                                const GPUMediumI& m,
                                //
                                const Surface& surface,         // Surface info (normals uvs etc.)
                                // Constants
                                const Data&,
                                const HitKey::Type& matId);

// For sampling radiance of direct wo - wi relation
template <class Data, class Surface>
using EvaluateFunc = Vector3(*)(// Input
                                const Vector3& wo,              // Outgoing Radiance
                                const Vector3& wi,              // Incoming Radiance
                                const Vector3& pos,             // Position
                                const GPUMediumI& m,
                                //
                                const Surface& surface,         // Surface info (normals uvs etc.)
                                // Constants
                                const Data&,
                                const HitKey::Type& matId);

// Checking PDF of such wo - wi relation
template <class Data, class Surface>
using PdfFunc = float(*)(// Input
                         const Vector3& wo,              // Outgoing Radiance
                         const Vector3& wi,              // Incoming Radiance
                         const Vector3& pos,             // Position
                         const GPUMediumI& m,
                         //
                         const Surface& surface,         // Surface info (normals uvs etc.)
                         // Constants
                         const Data&,
                         const HitKey::Type& matId);

// TEXTURE CACHING IS SHELVED

////===================================//
//// Texture Caching Related Functions //
////===================================//
//
//// This call is per-ray which returns multiple of UVs
//// that is dependant of the material.
////
//// Each ray returns its own UV location which may be required by the texture cache
////
//template <class Data, class Surface>
//using AcquireUVList = void(*)(//Output
//                              TexCoords*,
//                              const Surface& surface,
//                              // Constants
//                              const Data&,
//                              const HitKey::Type& matId);