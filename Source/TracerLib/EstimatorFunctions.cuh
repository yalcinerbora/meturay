#pragma once

#include  "RayLib/HybridFunctions.h"
#include "GPUEndpointI.cuh"
#include "Random.cuh"

// Basic Next Event Estimation
// Randomly choose a light and sample it
__device__
inline bool NextEventEstimation(HitKey& key,
                                uint32_t& lightIndex,
                                Vector3& direction,
                                float& lDistance,
                                float& pdf,
                                //
                                const Vector3& position,
                                RandomGPU& rng,
                                //
                                const GPULightI** gEndPoints,
                                const uint32_t pointCount)
{
    if(pointCount == 0) return false;

    // Randomly Select Light
    float r1 = GPUDistribution::Uniform<float>(rng);
    r1 *= static_cast<float>(pointCount);
    uint32_t index = static_cast<uint32_t>(floor(r1));
    
    // Extremely rarely index become the point count
    // although GPUDistribution::Uniform returns [0, 1)
    // it still happens due to fp error i guess?
    // if it happens just return the last light on the list
    if(index == pointCount) index--;

    const GPULightI* point = gEndPoints[index];
    point->Sample(lDistance, direction,
                  pdf, position, rng);
    // Incorporate the PDF of selecting that point
    pdf *= 1.0f / static_cast<float>(pointCount);
    lightIndex = index;
    key = point->BoundaryMaterial();
    return true;
}

// Basic Russian Roulette
__device__
inline bool RussianRoulette(Vector3& irradianceFactor,
                            float probFactor, RandomGPU& rng)
{
    // Basic Russian Roulette
    probFactor = HybridFuncs::Clamp(probFactor, 0.05f, 1.0f);
    if(GPUDistribution::Uniform<float>(rng) >= probFactor)
        return true;
    else irradianceFactor *= (1.0f / probFactor);
    return false;
}