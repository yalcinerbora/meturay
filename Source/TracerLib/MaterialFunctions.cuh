#pragma once

#include "RayLib/Ray.h"

class RandomGPU;

// Shade function is responsible for BRDF evalulation
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
using ShadeFunc = Vector3(*)(// Sampled Output
                             RayF& wo,
                             float& pdf,
                             // Input
                             const Vector3& wi,                // Incoming Radiance
                             const Vector3& pos,               // Position
                             const Surface& surface,
                             // I-O
                             RandomGPU& rng,
                             // Constants
                             const Data&,
                             const HitKey::Type& matId);

// Material provides this in order to for renderer to sample its own strategy
template <class Data, class Surface>
using EvaluateFunc = Vector3(*)(// Input
                                const Vector3& wo,                // Outgoing Radiance
                                const Vector3& wi,                // Incoming Radiance
                                const Vector3& pos,               // Position
                                const Surface& surface, 
                                // Constants
                                const Data&,
                                const HitKey::Type& matId);

// Add more if necessary