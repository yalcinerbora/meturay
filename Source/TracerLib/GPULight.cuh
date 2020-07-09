#pragma once

#include "GPUEndpointI.cuh"
#include "RayLib/Constants.h"

#include <type_traits>

#include "Random.cuh"
#include "RayLib/HybridFunctions.h"
#include "RayLib/Quaternion.h"


class PointLight final : public GPULightI
{
    private:
        // Sample Ready Parameters
        // All of which is world space
        Vector3             position;

    protected:
    public:
        // Constructors & Destructor
        __device__          PointLight(const Vector3& position,
                                       const Vector3& flux,
                                       HitKey k,
                                       PrimitiveId pId);

        // Interface 
        __device__ void     Sample(// Output
                                   HitKey& materialKey,
                                   PrimitiveId& primId,
                                   //
                                   float& distance,
                                   Vector3& direction,
                                   float& pdf,
                                   // Input
                                   const Vector3& position,
                                   // I-O
                                   RandomGPU&) const override;

        __device__ void     GenerateRay(// Output
                                        RayReg&,
                                        // Input
                                        const Vector2i& sampleId,
                                        const Vector2i& sampleMax,
                                        // I-O
                                        RandomGPU&) const override;

        __device__ Vector3  Flux(const Vector3& direction) const override;
};

class DirectionalLight final : public GPULightI
{
    private:
        // Sample Ready Parameters
        // All of which is world space
        Vector3             direction;

    protected:
    public:
        // Constructors & Destructor
        __device__          DirectionalLight(const Vector3& direction,
                                             const Vector3& flux,
                                             HitKey k,
                                             PrimitiveId pId);

        // Interface 
        __device__ void     Sample(// Output
                                   HitKey& materialKey,
                                   PrimitiveId& primId,
                                   //
                                   float& distance,
                                   Vector3& direction,
                                   float& pdf,
                                   // Input
                                   const Vector3& position,
                                   // I-O
                                   RandomGPU&) const override;

        __device__ void     GenerateRay(// Output
                                        RayReg&,
                                        // Input
                                        const Vector2i& sampleId,
                                        const Vector2i& sampleMax,
                                        // I-O
                                        RandomGPU&) const override;

        __device__ Vector3  Flux(const Vector3& direction) const override;
};

class SpotLight final : public GPULightI
{
    private:
        // Sample Ready Parameters
        // All of which is world space
        Vector3             position;
        float               cosMin;
        Vector3             direction;
        float               cosMax;

    protected:
    public:
        // Constructors & Destructor
        __device__          SpotLight(const Vector3& position,
                                      const Vector3& direction,
                                      const Vector2& coneMinMax, // Degrees
                                      const Vector3& flux,
                                      HitKey k,
                                      PrimitiveId pId);

        // Interface 
        __device__ void     Sample(// Output
                                   HitKey& materialKey,
                                   PrimitiveId& primId,
                                   //
                                   float& distance,
                                   Vector3& direction,
                                   float& pdf,
                                   // Input
                                   const Vector3& position,
                                   // I-O
                                   RandomGPU&) const override;

        __device__ void     GenerateRay(// Output
                                        RayReg&,
                                        // Input
                                        const Vector2i& sampleId,
                                        const Vector2i& sampleMax,
                                        // I-O
                                        RandomGPU&) const override;

        __device__ Vector3  Flux(const Vector3& direction) const override;
};

class RectangularLight final : public GPULightI
{
    private:
        // Sample Ready Parameters
        // All of which is world space
        Vector3             topLeft;
        Vector3             right;
        Vector3             down;
        Vector3             normal;
        float               area;

    protected:
    public:
        // Constructors & Destructor
        __device__          RectangularLight(const Vector3& topLeft,
                                             const Vector3& right,
                                             const Vector3& down,
                                             const Vector3& flux,
                                             HitKey k,
                                             PrimitiveId pId);

        // Interface 
        __device__ void     Sample(// Output
                                   HitKey& materialKey,
                                   PrimitiveId& primId,
                                   //
                                   float& distance,
                                   Vector3& direction,
                                   float& pdf,
                                   // Input
                                   const Vector3& position,
                                   // I-O
                                   RandomGPU&) const override;

        __device__ void     GenerateRay(// Output
                                        RayReg&,
                                        // Input
                                        const Vector2i& sampleId,
                                        const Vector2i& sampleMax,
                                        // I-O
                                        RandomGPU&) const override;

        __device__ Vector3  Flux(const Vector3& direction) const override;
};

class TriangularLight final : public GPULightI
{
    private:
        // Sample Ready Parameters
        // All of which is world space
        Vector3             v0;
        Vector3             v1;
        Vector3             v2;
        Vector3             normal;
        float               area;

    protected:
    public:
        // Constructors & Destructor
        __device__          TriangularLight(const Vector3& v0,
                                            const Vector3& v1,
                                            const Vector3& v2,
                                            const Vector3& flux,
                                            HitKey k,
                                            PrimitiveId pId);

        // Interface 
        __device__ void     Sample(// Output
                                   HitKey& materialKey,
                                   PrimitiveId& primId,
                                   //
                                   float& distance,
                                   Vector3& direction,
                                   float& pdf,
                                   // Input
                                   const Vector3& position,
                                   // I-O
                                   RandomGPU&) const override;

        __device__ void     GenerateRay(// Output
                                        RayReg&,
                                        // Input
                                        const Vector2i& sampleId,
                                        const Vector2i& sampleMax,
                                        // I-O
                                        RandomGPU&) const override;

        __device__ Vector3  Flux(const Vector3& direction) const override;
};

class DiskLight final : public GPULightI
{
    private:
        // Sample Ready Parameters
        // All of which is world space
        Vector3             center;
        Vector3             normal;
        float               radius;
        float               area;

    protected:
    public:
        // Constructors & Destructor
        __device__          DiskLight(const Vector3& center,
                                      const Vector3& normal,
                                      float radius,
                                      const Vector3& flux,
                                      HitKey k,
                                      PrimitiveId pId);

        // Interface 
        __device__ void     Sample(// Output
                                   HitKey& materialKey,
                                   PrimitiveId& primId,
                                   //
                                   float& distance,
                                   Vector3& direction,
                                   float& pdf,
                                   // Input
                                   const Vector3& position,
                                   // I-O
                                   RandomGPU&) const override;

        __device__ void     GenerateRay(// Output
                                        RayReg&,
                                        // Input
                                        const Vector2i& sampleId,
                                        const Vector2i& sampleMax,
                                        // I-O
                                        RandomGPU&) const override;

        __device__ Vector3  Flux(const Vector3& direction) const override;
};

class SphericalLight final : public GPULightI
{
    private:
        // Sample Ready Parameters
        // All of which is world space
        Vector3             center;
        float               radius;
        float               area;

    protected:
    public:
        // Constructors & Destructor
        __device__          SphericalLight(const Vector3& center,
                                           float radius,
                                           const Vector3& flux,
                                           HitKey k,
                                           PrimitiveId pId);

        // Interface 
        __device__ void     Sample(// Output
                                   HitKey& materialKey,
                                   PrimitiveId& primId,
                                   //
                                   float& distance,
                                   Vector3& direction,
                                   float& pdf,
                                   // Input
                                   const Vector3& position,
                                   // I-O
                                   RandomGPU&) const override;

        __device__ void     GenerateRay(// Output
                                        RayReg&,
                                        // Input
                                        const Vector2i& sampleId,
                                        const Vector2i& sampleMax,
                                        // I-O
                                        RandomGPU&) const override;

        __device__ Vector3  Flux(const Vector3& direction) const override;
};

// Expand this when necessary
static constexpr size_t LightSizeArray[] = {sizeof(PointLight), sizeof(DirectionalLight),
                                            sizeof(SpotLight),sizeof(RectangularLight),
                                            sizeof(TriangularLight),sizeof(DiskLight),
                                            sizeof(SphericalLight)};
static constexpr size_t GPULightUnionSize = *std::min_element(std::begin(LightSizeArray),
                                                               std::end(LightSizeArray));

__device__
inline PointLight::PointLight(const Vector3& position,
                              const Vector3& flux,
                              HitKey k,
                              PrimitiveId pId)
    : GPULightI(flux, k, pId)
    , position(position)
{}

__device__
inline DirectionalLight::DirectionalLight(const Vector3& direction,
                                          const Vector3& flux,
                                          HitKey k,
                                          PrimitiveId pId)
    : GPULightI(flux, k, pId)
    , direction(direction)
{}

__device__
inline SpotLight::SpotLight(const Vector3& position,
                            const Vector3& direction,
                            const Vector2& coneMinMax, // Degrees
                            const Vector3& flux,
                            HitKey k,
                            PrimitiveId pId)
    : GPULightI(flux, k, pId)
    , position(position)
    , direction(direction)
    , cosMin(coneMinMax[0])
    , cosMax(coneMinMax[1])
{}

__device__
inline RectangularLight::RectangularLight(const Vector3& topLeft,
                                          const Vector3& right,
                                          const Vector3& down,
                                          const Vector3& flux,
                                          HitKey k,
                                          PrimitiveId pId)
    : GPULightI(flux, k, pId)
    , topLeft(topLeft)
    , right(right)
    , down(down)
{
    Vector3 cross = Cross(down, right);
    area = cross.Length();
    normal = cross.Normalize();
}

__device__
inline TriangularLight::TriangularLight(const Vector3& v0,
                                        const Vector3& v1,
                                        const Vector3& v2,
                                        const Vector3& flux,
                                        HitKey k,
                                        PrimitiveId pId)
    : GPULightI(flux, k, pId)
    , v0(v0)
    , v1(v1)
    , v2(v2)
{
    // CCW Triangle
    Vector3 cross = Cross((v1 - v0), (v2 - v0));
    area = 0.5f * cross.Length();
    normal = cross.Normalize();
}

__device__
inline DiskLight::DiskLight(const Vector3& center,
                            const Vector3& normal,
                            float radius,
                            const Vector3& flux,
                            HitKey k,
                            PrimitiveId pId)
    : GPULightI(flux, k, pId)
    , center(center)
    , normal(normal)
    , radius(radius)
    , area(MathConstants::Pi* radius* radius)
{}

__device__
inline SphericalLight::SphericalLight(const Vector3& center,
                                      float radius,
                                      const Vector3& flux,
                                      HitKey k,
                                      PrimitiveId pId)
    : GPULightI(flux, k, pId)
    , center(center)
    , radius(radius)
    , area (MathConstants::Pi * radius * radius * 4.0f)
{}

__device__
inline void PointLight::Sample(// Output
                               HitKey& materialKey,
                               PrimitiveId& primId,
                               //
                               float& distance,
                               Vector3& direction,
                               float& pdf,
                               // Input
                               const Vector3& worldLoc,
                               // I-O
                               RandomGPU&) const
{
    materialKey = boundaryMaterialKey;
    primId = primitiveId;

    direction = (position - worldLoc);
    distance = direction.Length();
    direction.NormalizeSelf();
    pdf = 1.0f;
}

__device__
inline void PointLight::GenerateRay(// Output
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
inline Vector3 PointLight::Flux(const Vector3& direction) const
{
    return flux;
}

// ========================================= 
__device__
inline void DirectionalLight::Sample(// Output
                                     HitKey& materialKey,
                                     PrimitiveId& primId,
                                     //
                                     float& distance,
                                     Vector3& dir,
                                     float& pdf,
                                     // Input
                                     const Vector3& worldLoc,
                                     // I-O
                                     RandomGPU&) const
{
    materialKey = boundaryMaterialKey;
    primId = primitiveId;

    dir = -direction;
    distance = FLT_MAX;
    pdf = 1.0f;
}

__device__
inline void DirectionalLight::GenerateRay(// Output
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
inline Vector3 DirectionalLight::Flux(const Vector3& dir) const
{
    static constexpr Vector3 ZERO = Zero3;
    return (dir.Dot(direction) < 0.0f) ? ZERO : flux;
}

// ========================================= 
__device__
inline void SpotLight::Sample(// Output
                              HitKey& materialKey,
                              PrimitiveId& primId,
                              //
                              float& distance,
                              Vector3& dir,
                              float& pdf,
                              // Input
                              const Vector3& worldLoc,
                              // I-O
                              RandomGPU&) const
{
    materialKey = boundaryMaterialKey;
    primId = primitiveId;

    dir = -direction;
    distance = (position - worldLoc).Length();
    pdf = 1.0f;
}

__device__ void
inline SpotLight::GenerateRay(// Output
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
inline Vector3 SpotLight::Flux(const Vector3& dir) const
{
    float cos = HybridFuncs::Clamp(dir.Dot(direction), cosMin, cosMax);
    float factor = (cos - cosMin) / (cosMax - cosMin);
    factor *= factor;
    return flux * factor * factor;
}

// ========================================= 
__device__
inline void RectangularLight::Sample(// Output
                                     HitKey& materialKey,
                                     PrimitiveId& primId,
                                     //
                                     float& distance,
                                     Vector3& direction,
                                     float& pdf,
                                     // Input
                                     const Vector3& worldLoc,
                                     // I-O
                                     RandomGPU& rng) const
{
    materialKey = boundaryMaterialKey;
    primId = primitiveId;

    float x = GPUDistribution::Uniform<float>(rng);
    float y = GPUDistribution::Uniform<float>(rng);
    Vector3 position = topLeft + right * x + down * y;

    float nDotL = max(normal.Dot(-direction), 0.0f);
    direction = position - worldLoc;
    distance = direction.LengthSqr();
    pdf = distance / (nDotL * area);
    distance = sqrt(distance);
    direction.NormalizeSelf();
}

__device__
inline void RectangularLight::GenerateRay(// Output
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
inline Vector3 RectangularLight::Flux(const Vector3& dir) const
{
    static constexpr Vector3 ZERO = Zero3;
    return (dir.Dot(normal) < 0.0f) ? ZERO : flux;
}

// ========================================= 
__device__
inline void TriangularLight::Sample(// Output
                                    HitKey& materialKey,
                                    PrimitiveId& primId,
                                    //
                                    float& distance,
                                    Vector3& direction,
                                    float& pdf,
                                    // Input
                                    const Vector3& worldLoc,
                                    // I-O
                                    RandomGPU& rng) const
{
    materialKey = boundaryMaterialKey;
    primId = primitiveId;

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
    distance = direction.LengthSqr();
    pdf = distance / (nDotL * area);
    distance = sqrt(distance);
    direction.NormalizeSelf();
}

__device__
inline void TriangularLight::GenerateRay(// Output
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
inline Vector3 TriangularLight::Flux(const Vector3& dir) const
{
    static constexpr Vector3 ZERO = Zero3;
    return (dir.Dot(normal) < 0.0f) ? ZERO : flux;
}

// ========================================= 
__device__
inline void DiskLight::Sample(// Output
                              HitKey& materialKey,
                              PrimitiveId& primId,
                              //
                              float& distance,
                              Vector3& direction,
                              float& pdf,
                              // Input
                              const Vector3& worldLoc,
                              // I-O
                              RandomGPU& rng) const
{
    materialKey = boundaryMaterialKey;
    primId = primitiveId;

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
    distance = direction.LengthSqr();
    pdf = distance / (nDotL * area);
    distance = sqrt(distance);
    direction.NormalizeSelf();
}

__device__
inline void DiskLight::GenerateRay(// Output
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
inline Vector3 DiskLight::Flux(const Vector3& dir) const
{
    static constexpr Vector3 ZERO = Zero3;
    return (dir.Dot(normal) < 0.0f) ? ZERO : flux;
}

// ========================================= 
__device__
inline void SphericalLight::Sample(// Output
                                   HitKey& materialKey,
                                   PrimitiveId& primId,
                                   //
                                   float& distance,
                                   Vector3& direction,
                                   float& pdf,
                                   // Input
                                   const Vector3& worldLoc,
                                   // I-O
                                   RandomGPU& rng) const
{
    materialKey = boundaryMaterialKey;
    primId = primitiveId;

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
    distance = direction.LengthSqr();
    pdf = distance / area;
    distance = sqrt(distance);
    direction.NormalizeSelf();
}

__device__
inline void SphericalLight::GenerateRay(// Output
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
inline Vector3 SphericalLight::Flux(const Vector3& direction) const
{
    return flux;
}


