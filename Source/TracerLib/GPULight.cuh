#pragma once

#include "GPUEndpointI.cuh"
#include "RayLib/Constants.h"

#include <type_traits>

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
                                       HitKey k);

        // Interface 
        __device__ void     Sample(// Output
                                   HitKey& materialKey,
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
                                             HitKey k);

        // Interface 
        __device__ void     Sample(// Output
                                   HitKey& materialKey,
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
                                      HitKey k);

        // Interface 
        __device__ void     Sample(// Output
                                   HitKey& materialKey,
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
                                             HitKey k);

        // Interface 
        __device__ void     Sample(// Output
                                   HitKey& materialKey,
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
                                            HitKey k);

        // Interface 
        __device__ void     Sample(// Output
                                   HitKey& materialKey,
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
                                      HitKey k);

        // Interface 
        __device__ void     Sample(// Output
                                   HitKey& materialKey,
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
                                           HitKey k);

        // Interface 
        __device__ void     Sample(// Output
                                   HitKey& materialKey,
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

static constexpr size_t GPULightUnionSize = std::aligned_union<7,
                                                               PointLight, DirectionalLight,
                                                               SpotLight, RectangularLight,
                                                               TriangularLight, DiskLight,
                                                               SphericalLight>::alignment_value;

__device__
inline PointLight::PointLight(const Vector3& position,
                              const Vector3& flux,
                              HitKey k)
    : GPULightI(flux, k)
    , position(position)
{}

__device__
inline DirectionalLight::DirectionalLight(const Vector3& direction,
                                          const Vector3& flux,
                                          HitKey k)
    : GPULightI(flux, k)
    , direction(direction)
{}

__device__
inline SpotLight::SpotLight(const Vector3& position,
                            const Vector3& direction,
                            const Vector2& coneMinMax, // Degrees
                            const Vector3& flux,
                            HitKey k)
    : GPULightI(flux, k)
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
                                          HitKey k)
    : GPULightI(flux, k)
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
                                        HitKey k)
    : GPULightI(flux, k)
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
                            HitKey k)
    : GPULightI(flux, k)
    , center(center)
    , normal(normal)
    , radius(radius)
    , area(MathConstants::Pi* radius* radius)
{}

__device__
inline SphericalLight::SphericalLight(const Vector3& center,
                                      float radius,
                                      const Vector3& flux,
                                      HitKey k)
    : GPULightI(flux, k)
    , center(center)
    , radius(radius)
    , area (MathConstants::Pi * radius * radius * 4.0f)
{}


