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
    uint32_t index = static_cast<uint32_t>(round(r1));

    const GPUEndpointI* point = gEndPoints[index];
    point->Sample(key, lDistance, direction,
                  pdf, position, rng);
    // Incorporate the PDF of selecting that point
    pdf *= 1.0f / static_cast<float>(pointCount);
    lightIndex = index;
    return true;
}

// Basic Russian Roulette
__device__
inline bool RussianRoulette(Vector3& irradianceFactor,
                            float probFactor, RandomGPU& rng)
{
    // Basic Russian Roulette
    probFactor = HybridFuncs::Clamp(probFactor, 0.0f, 1.0f);
    if(GPUDistribution::Uniform<float>(rng) >= probFactor)
        return true;
    else irradianceFactor *= (1.0f / probFactor);
    return false;
}