#pragma once

#include "EstimatorFunctions.cuh"
#include "RayLib/HybridFunctions.h"

struct EmptyEstimatorData {};

struct BasicEstimatorData 
{
    const EstimatorInfo* dLights;
    uint32_t lightCount;
};

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
inline bool EstimateEventBasic(HitKey& key,
                               Vector3& direction,
                               float& pdf,
                               //
                               const Vector3& irradianceFactor,
                               const Vector3& position,
                               RandomGPU& rng,
                               //
                               const BasicEstimatorData& estData)
{
    if(estData.lightCount == 0) return false;

    // Randomly Select Light
    float r1 = GPUDistribution::Uniform<float>(rng);
    r1 *= static_cast<float>(estData.lightCount);
    uint32_t lightIndex = static_cast<uint32_t>(round(r1));
    EstimatorInfo info = estData.dLights[lightIndex];

    // TODO: Implement All
    // Just do triangle for now
    if(info.type == LightType::TRIANGULAR)
    {
        Vector3 p0 = info.position0R;
        Vector3 p1 = info.position1G;
        Vector3 p2 = info.position2B;
        // Sample a point on tri
        Vector3 triPoint = SampleTriangle(rng, p0, p1, p2);        
   
        // Calc Direction
        direction = (triPoint - position).Normalize();
        // Push Ray
        key = info.matKey;

        Vector3 cross = Cross((p1 - p0), (p2 - p0));
        float triArea = 0.5f * cross.Length();
        Vector3 normal = cross.Normalize();
        
        //float nDotL = abs(normal.Dot(-direction));
        float nDotL = max(normal.Dot(-direction), 0.0f);

        // From Pbrt Book
        // PDF of the light on the triangle
        pdf = (position - triPoint).LengthSqr() /
              ((nDotL) * triArea);
        // PDF of selecting that light
        pdf *= 1.0f / static_cast<float>(estData.lightCount);
        return true;
    }
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