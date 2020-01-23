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

    // Initial PDF is always true (For Delta Distrib)
    pdf = 1.0f;
    float tMax;

    // Randomly Select Light
    float r1 = GPUDistribution::Uniform<float>(rng);
    r1 *= static_cast<float>(estData.lightCount);
    uint32_t lightIndex = static_cast<uint32_t>(round(r1));
    EstimatorInfo info = estData.dLights[lightIndex];

    
    Vector3 normal;
    Vector3 samplePoint;
    float area = 0.0f;    
    switch(info.type)
    {
        case LightType::POINT:
        {
            samplePoint = SamplePoint(rng, info.position0R);
            break;
        }
        case LightType::DIRECTIONAL:
        {
            direction = info.position0R;
            tMax = INFINITY;
            break;
        }
        case LightType::SPOT:
        {
            // TODO:: Implement
            return false;
            break;
        }
        case LightType::RECTANGULAR:
        {
            Vector3 v0 = Vector3(info.position1G);
            Vector3 v1 = Vector3(info.position2B);
            samplePoint = SampleParallelogram(rng,
                                              info.position0R,
                                              v0, v1);

            Vector3 cross = Cross(v1, v0);
            area =  cross.Length();
            normal = cross.Normalize();            
            break;
        }
        case LightType::TRIANGULAR:
        {
            Vector3 p0 = info.position0R;
            Vector3 p1 = info.position1G;
            Vector3 p2 = info.position2B;
            // Sample a point on tri
            samplePoint = SampleTriangle(rng, p0, p1, p2);

            // Gen Triangle Area And Normal
            Vector3 cross = Cross((p1 - p0), (p2 - p0));
            area = 0.5f * cross.Length();
            normal = cross.Normalize();
            break;
        }
        case LightType::DISK:
        {
            Vector3 center = info.position0R;
            normal = info.position1G;
            float radius = info.position2B[0];
            samplePoint = SampleDisk(rng, center,
                                     normal, radius);
            area = MathConstants::Pi * radius * radius;            
            break;
        }
        case LightType::SPHERICAL:
        {
            // TODO:: Implement
            return false;
            break;
        }
        default: return false;
    }
           
    // If Area Light Gen PDF Acordignly
    if(info.type == LightType::RECTANGULAR ||
       info.type == LightType::TRIANGULAR ||
       info.type == LightType::DISK || 
       info.type == LightType::POINT)
    {
        // Calc Direction
        direction = (samplePoint - position);
        tMax = direction.Length();
        direction *= (1.0f / tMax);
    }
    if(info.type == LightType::RECTANGULAR ||
       info.type == LightType::TRIANGULAR ||
       info.type == LightType::DISK)
    {
        //float nDotL = abs(normal.Dot(-direction));
        float nDotL = max(normal.Dot(-direction), 0.0f);

        // From Pbrt Book
        // PDF of an area light 
        pdf = (position - samplePoint).LengthSqr() /
               ((nDotL) * area);
    }

    // Put Key
    key = info.matKey;
    
    // PDF of selecting that light
    pdf *= 1.0f / static_cast<float>(estData.lightCount);
    return true;
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