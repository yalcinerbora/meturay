#pragma once

#include "EstimatorFunctions.cuh"
#include "RayLib/HybridFunctions.h"

struct EmptyEstimatorData {};

struct BasicEstimatorData {};

__device__
inline bool TerminateEventBasic(Vector3& irradianceFactor,
                                float probFactor, RandomGPU& rng)
{
    // Basic Russian Roulette
    probFactor = HybridFuncs::Clamp(probFactor, 0.0f, 1.0f);
    if(GPUDistribution::Uniform<float>(rng) >= probFactor)
        return true;
    else irradianceFactor *= 1.0f / probFactor;
    return false;
}

__device__
inline bool TerminateEventEmpty(Vector3& irradianceFactor, 
                                float factor, RandomGPU& rng)
{
    return false;
}

__device__
inline bool EstimateEventEmpty(HitKey&,
                               Vector3&,
                               float&,
                               //
                               const Vector3&,
                               const Vector3&,
                               RandomGPU&,
                               //
                               const EmptyEstimatorData&)
{
    return false;
}

__device__
inline bool EstimateEventBasic(HitKey&,
                               Vector3&,
                               float&,
                               //
                               const Vector3&,
                               const Vector3&,
                               RandomGPU&,
                               //
                               const BasicEstimatorData&)
{
    return false;
}