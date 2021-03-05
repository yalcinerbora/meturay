#pragma once

#include  "RayLib/HybridFunctions.h"
#include "GPUEndpointI.h"
#include "Random.cuh"

// Basic Next Event Estimation
// Randomly choose a light and sample it
__device__
inline bool DoNextEventEstimation(HitKey& key,
                                  uint32_t& lightIndex,
                                  Vector3& direction,
                                  float& lDistance,
                                  float& pdf,
                                  //
                                  const Vector3& position,
                                  RandomGPU& rng,
                                  //
                                  const GPULightI** gLights,
                                  const uint32_t lightCount)
{
    if(lightCount == 0) return false;

    // Randomly Select Light
    float r1 = GPUDistribution::Uniform<float>(rng);
    r1 *= static_cast<float>(lightCount);
    uint32_t index = static_cast<uint32_t>(r1);
    
    // Extremely rarely index becomes the ligjt count
    // although GPUDistribution::Uniform should return [0, 1)
    // it still happens due to fp error i guess?
    // if it happens just return the last light on the list
    if(index == lightCount) index--;

    //printf("NEE Index %u total lights %u\n", index, lightCount);

    const GPULightI* light = gLights[index];
    light->Sample(lDistance, direction,
                  pdf, position, rng);
    // Incorporate the PDF of selecting that ligjt
    pdf *= (1.0f / static_cast<float>(lightCount));
    lightIndex = index;
    key = light->BoundaryMaterial();
    return true;
}

// Basic Russian Roulette
__device__
inline bool RussianRoulette(Vector3& irradianceFactor,
                            float probFactor, RandomGPU& rng)
{
    // Basic Russian Roulette
    probFactor = HybridFuncs::Clamp(probFactor, 0.005f, 1.0f);
    if(GPUDistribution::Uniform<float>(rng) >= probFactor)
        return true;
    else irradianceFactor *= (1.0f / probFactor);
    return false;
}