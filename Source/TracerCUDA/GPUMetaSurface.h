#pragma once

#include "RayLib/HitStructs.h"
#include "RayLib/Quaternion.h"
#include "RayLib/Ray.h"

#include "GPUSurface.h"
#include "GPUMaterialI.h"

class RNGeneratorGPUI;
class GPUTransformI;
class GPUMediumI;

// Surface Class
// Primitive Type / Material Type Agnostic
// beacuse of that it does not have full functionality
// of a templated MaterialGroup / PrimtiveGroup / WorkGroup
// triplet
// This is created so that a non workgroup related kernel can
// do shading calculations if needed
// (In our case WFPG Tracer uses this to access per-ray shading data)
// and do product path guiding
class GPUMetaSurface
{
    private:
    const GPUTransformI&        t;              // local to world

    UVSurface                   uvSurf;
    const GPUMaterialI*         gMaterial;

    public:
                                GPUMetaSurface(const GPUTransformI&,
                                               const UVSurface& uvSurface,
                                               const GPUMaterialI* gMaterial);

    // Normal Stuff
    __device__ Vector3f     WorldNormal() const;
    __device__ Vector3f     WorldGeoNormal() const;
    __device__ Vector3f     WorldPosition() const;
    //
    __device__ bool         IsEmissive() const;
    __device__ bool         Specularity() const;
    __device__ Vector3f     Sample(// Sampled Output
                                   RayF& wo,                       // Out direction
                                   float& pdf,                     // PDF for Monte Carlo
                                   const GPUMediumI*& outMedium,
                                   // Input
                                   const Vector3& wi,              // Incoming Radiance
                                   const Vector3& pos,             // Position
                                   const GPUMediumI& m,
                                   // I-O
                                   RNGeneratorGPUI& rng) const;
    __device__ Vector3f     Emit(// Input
                                 const Vector3& wo,      // Outgoing Radiance
                                 const Vector3& pos,     // Position
                                 const GPUMediumI& m) const;
    __device__ Vector3f     Evaluate(// Input
                                     const Vector3& wo,              // Outgoing Radiance
                                     const Vector3& wi,              // Incoming Radiance
                                     const Vector3& pos,             // Position
                                     const GPUMediumI& m) const;

    __device__ float        Pdf(// Input
                                const Vector3& wo,      // Outgoing Radiance
                                const Vector3& wi,
                                const Vector3& pos,     // Position
                                const GPUMediumI& m);
};

__device__ inline
Vector3f GPUMetaSurface::WorldNormal() const
{
    return uvSurf.WorldNormal();
}

__device__ inline
Vector3f GPUMetaSurface::WorldGeoNormal() const
{

    return uvSurf.WorldGeoNormal();
}

__device__ inline
Vector3f GPUMetaSurface::WorldPosition() const
{
    return uvSurf.WorldPosition();
}

__device__ inline
bool GPUMetaSurface::IsEmissive() const
{
    return gMaterial->IsEmissive();
}

__device__ inline
bool GPUMetaSurface::Specularity() const
{
    return gMaterial->Specularity(uvSurf);
}

__device__ inline
Vector3f GPUMetaSurface::Sample(// Sampled Output
                                RayF& wo,                       // Out direction
                                float& pdf,                     // PDF for Monte Carlo
                                const GPUMediumI*& outMedium,
                                // Input
                                const Vector3& wi,              // Incoming Radiance
                                const Vector3& pos,             // Position
                                const GPUMediumI& m,
                                // I-O
                                RNGeneratorGPUI& rng) const
{
    return gMaterial->Sample(wo, pdf, outMedium,
                             wi, pos, m, uvSurf,
                             rng);
}
__device__ inline
Vector3f GPUMetaSurface::Emit(const Vector3& wo,
                              const Vector3& pos,
                              const GPUMediumI& m) const
{
    return gMaterial->Emit(wo, pos, m, uvSurf);
}

__device__ inline
Vector3f GPUMetaSurface::Evaluate(const Vector3& wo,
                                  const Vector3& wi,
                                  const Vector3& pos,
                                  const GPUMediumI& m) const
{
    return gMaterial->Evaluate(wo, wi, pos, m, uvSurf);
}

__device__ inline
float GPUMetaSurface::Pdf(const Vector3& wo,
                          const Vector3& wi,
                          const Vector3& pos,
                          const GPUMediumI& m)
{
    return gMaterial->Pdf(wo, wi, pos, m, uvSurf);
}