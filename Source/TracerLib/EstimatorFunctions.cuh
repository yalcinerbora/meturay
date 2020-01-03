#pragma once

#include "RayLib/HitStructs.h"
#include "RayLib/Vector.h"
#include "RayLib/SceneStructs.h"

#include "Random.cuh"

template <class EstimatorData>
using EstimateEventFunc = bool(*)(// Output
                                  HitKey& boundaryMatKey,
                                  Vector3& direction,
                                  float& probability,
                                  // Input
                                  const Vector3& irradianceFactor,
                                  const Vector3& position,
                                  RandomGPU& rng,
                                  //
                                  const EstimatorData&);


// Sampling Functions
__device__
inline Vector3 SampleTriangle(RandomGPU& rng,
                              // Primitive Specific Data
                              const Vector3& p0,
                              const Vector3& p1,
                              const Vector3& p2)
{
    float r1 = sqrt(GPUDistribution::Uniform<float>(rng));
    float r2 = GPUDistribution::Uniform<float>(rng);

    // Osada 2002
    // http://graphics.stanford.edu/courses/cs468-08-fall/pdf/osada.pdf
    float a = 1 - r1;
    float b = (1 - r2) * r1;
    float c = r1 * r2;

    return (p0 * a + p1 * b + p2 * c);
}

__device__
inline Vector3 SampleSphere(RandomGPU& rng,
                            // Primitive Specific Data
                            const Vector3& center,
                            float radius)
{
    // Marsaglia 1972
    // http://mathworld.wolfram.com/SpherePointPicking.html
    
    float x1 = GPUDistribution::Uniform<float>(rng) * 2.0f - 1.0f;
    float x2 = GPUDistribution::Uniform<float>(rng) * 2.0f - 1.0f;

    float x1Sqr = x1 * x1;
    float x2Sqr = x2 * x2;
    float coeff = sqrt(1 - x1Sqr - x2Sqr);

    Vector3 unitSphr = Vector3(2.0f * x1 * coeff,
                               2.0f * x2 * coeff,
                               1.0f - 2.0f * (x1Sqr + x2Sqr));

    return center + radius * unitSphr;
}

__device__
inline Vector3 SampleDisk(RandomGPU& rng,
                          // Primitive Specific Data
                          const Vector3& center,
                          const Vector3& normal,
                          float radius)
{

    float r = GPUDistribution::Uniform<float>(rng) * radius;
    float tetha = GPUDistribution::Uniform<float>(rng) * 2.0f * MathConstants::Pi;

    // Aligned to Axis Z
    Vector3 disk = Vector3(sqrt(r) * cos(tetha),
                               sqrt(r) * sin(tetha),
                               0.0f);               
    // Rotate to disk normal
    QuatF rotation = QuatF::RotationBetweenZAxis(normal);
    Vector3 worldDisk = rotation.ApplyRotation(disk);

    return center + worldDisk;
}

__device__
inline Vector3 SampleParallelogram(RandomGPU& rng,
                                   // Primitive Specific Data
                                   const Vector3& topLeft,
                                   const Vector3& v0,
                                   const Vector3& v1,
                                   const Vector2& length)
{
    float x = GPUDistribution::Uniform<float>(rng) * length[0];
    float y = GPUDistribution::Uniform<float>(rng) * length[1];
    Vector3 location = v0 * x + v1 * y;
    return topLeft + location;
}

__device__
inline Vector3 SamplePoint(RandomGPU& rng,
                           // Primitive Specific Data
                           const Vector3& point)
{
    // Singularity; you cant uniform sample
    return point;
}