#include "GPULight.cuh"
#include "Random.cuh"
#include "RayLib/HybridFunctions.h"
#include "RayLib/Quaternion.h"

__device__
void PointLight::Sample(// Output
                        HitKey& materialKey,
                        Vector3& direction,
                        float& pdf,
                        // Input
                        const Vector3& worldLoc,
                        // I-O
                        RandomGPU&) const
{
    materialKey = boundaryMaterialKey;
    direction = position;
    pdf = 1.0f;
}

__device__
void PointLight::GenerateRay(// Output
                             RayReg&,
                             // Input
                             const Vector2i& sampleId,
                             const Vector2i& sampleMax,
                             // I-O
                             RandomGPU&) const
{
    // TODO:
}

__device__ 
Vector3 PointLight::Flux(const Vector3& direction) const
{
    return flux;
}

// ========================================= 
__device__
void DirectionalLight::Sample(// Output
                              HitKey& materialKey,
                              Vector3& dir,
                              float& pdf,
                              // Input
                              const Vector3& worldLoc,
                              // I-O
                              RandomGPU&) const
{
    materialKey = boundaryMaterialKey;
    dir = -direction;
    pdf = 1.0f;
}

__device__
void DirectionalLight::GenerateRay(// Output
                                   RayReg&,
                                   // Input
                                   const Vector2i& sampleId,
                                   const Vector2i& sampleMax,
                                   // I-O
                                   RandomGPU&) const
{
    // TODO:
}

__device__ 
Vector3 DirectionalLight::Flux(const Vector3& dir) const
{
    return (dir.Dot(direction) < 0.0f) ? Zero3 : flux;
}

// ========================================= 
__device__
void SpotLight::Sample(// Output
                       HitKey& materialKey,
                       Vector3& dir,
                       float& pdf,
                       // Input
                       const Vector3& worldLoc,
                       // I-O
                       RandomGPU&) const
{
    materialKey = boundaryMaterialKey;
    dir = -direction;
    pdf = 1.0f;
}

__device__ void
SpotLight::GenerateRay(// Output
                       RayReg&,
                       // Input
                       const Vector2i& sampleId,
                       const Vector2i& sampleMax,
                       // I-O
                       RandomGPU&) const
{
    // TODO:

}

__device__ 
Vector3 SpotLight::Flux(const Vector3& dir) const
{
    float cos = HybridFuncs::Clamp(dir.Dot(direction), cosMin, cosMax);
    float factor = (cos - cosMin) / (cosMax - cosMin);
    factor *= factor;
    return flux * factor * factor;
}

// ========================================= 
__device__
void RectangularLight::Sample(// Output
                              HitKey& materialKey,
                              Vector3& direction,
                              float& pdf,
                              // Input
                              const Vector3& worldLoc,
                              // I-O
                              RandomGPU& rng) const
{
    materialKey = boundaryMaterialKey;
    float x = GPUDistribution::Uniform<float>(rng);
    float y = GPUDistribution::Uniform<float>(rng);
    Vector3 position =  topLeft + right * x + down * y;

    float nDotL = max(normal.Dot(-direction), 0.0f);
    direction = position - worldLoc;
    pdf = direction.LengthSqr() / (nDotL * area);
}

__device__
void RectangularLight::GenerateRay(// Output
                                   RayReg&,
                                   // Input
                                   const Vector2i& sampleId,
                                   const Vector2i& sampleMax,
                                   // I-O
                                   RandomGPU&) const
{
    // TODO:
}

__device__ 
Vector3 RectangularLight::Flux(const Vector3& dir) const
{
    return (dir.Dot(normal) < 0.0f) ? Zero3 : flux;
}

// ========================================= 
__device__
void TriangularLight::Sample(// Output
                             HitKey& materialKey,
                             Vector3& direction,
                             float& pdf,
                             // Input
                             const Vector3& worldLoc,
                             // I-O
                             RandomGPU& rng) const
{
    materialKey = boundaryMaterialKey;

    float r1 = sqrt(GPUDistribution::Uniform<float>(rng));
    float r2 = GPUDistribution::Uniform<float>(rng);

    // Osada 2002
    // http://graphics.stanford.edu/courses/cs468-08-fall/pdf/osada.pdf
    float a = 1 - r1;
    float b = (1 - r2) * r1;
    float c = r1 * r2;

    Vector3 position = (v0 * a + v1 * b + v2 * c);

    float nDotL = max(normal.Dot(-direction), 0.0f);
    direction = position - worldLoc;
    pdf = direction.LengthSqr() / (nDotL * area);
}

__device__
void TriangularLight::GenerateRay(// Output
                                  RayReg&,
                                  // Input
                                  const Vector2i& sampleId,
                                  const Vector2i& sampleMax,
                                  // I-O
                                  RandomGPU&) const
{
    // TODO:
}

__device__ 
Vector3 TriangularLight::Flux(const Vector3& dir) const
{
    return (dir.Dot(normal) < 0.0f) ? Zero3 : flux;
}

// ========================================= 
__device__
void DiskLight::Sample(// Output
                       HitKey& materialKey,
                       Vector3& direction,
                       float& pdf,
                       // Input
                       const Vector3& worldLoc,
                       // I-O
                       RandomGPU& rng) const
{
    materialKey = boundaryMaterialKey;

    float r = GPUDistribution::Uniform<float>(rng) * radius;
    float tetha = GPUDistribution::Uniform<float>(rng) * 2.0f * MathConstants::Pi;

    // Aligned to Axis Z
    Vector3 disk = Vector3(sqrt(r) * cos(tetha),
                               sqrt(r) * sin(tetha),
                               0.0f);               
    // Rotate to disk normal
    QuatF rotation = QuatF::RotationBetweenZAxis(normal);
    Vector3 worldDisk = rotation.ApplyRotation(disk);
    Vector3 position = center + worldDisk;

    float nDotL = max(normal.Dot(-direction), 0.0f);
    direction = position - worldLoc;
    pdf = direction.LengthSqr() / (nDotL * area);
}

__device__
void DiskLight::GenerateRay(// Output
                            RayReg&,
                            // Input
                            const Vector2i& sampleId,
                            const Vector2i& sampleMax,
                            // I-O
                            RandomGPU&) const
{
    // TODO:
}

__device__ 
Vector3 DiskLight::Flux(const Vector3& dir) const
{
    return (dir.Dot(normal) < 0.0f) ? Zero3 : flux;
}

// ========================================= 
__device__
void SphericalLight::Sample(// Output
                            HitKey& materialKey,
                            Vector3& direction,
                            float& pdf,
                            // Input
                            const Vector3& worldLoc,
                            // I-O
                            RandomGPU& rng) const
{
    materialKey = boundaryMaterialKey;

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

    Vector3 position = center + radius * unitSphr;
    direction = position - worldLoc;
    pdf = direction.LengthSqr() /  area;
}

__device__
void SphericalLight::GenerateRay(// Output
                                 RayReg&,
                                 // Input
                                 const Vector2i& sampleId,
                                 const Vector2i& sampleMax,
                                 // I-O
                                 RandomGPU&) const
{
    // TODO
}

__device__ 
Vector3 SphericalLight::Flux(const Vector3& direction) const
{
    return flux;
}