#pragma once

struct RayReg;
class RandomGPU;

#include "RayAuxStruct.h"
#include "MaterialDataStructs.h"
#include "SurfaceStructs.h"

#include "RayLib/Constants.h"

#include "TracerLib/ImageFunctions.cuh"
#include "TracerLib/GPUEventEstimatorEmpty.h"


Vector3 ConstantShade(// Sampled Output
                     RayF& wo,
                     float& pdf,
                     // Input
                     const Vector3& wi,
                     const Vector3& pos,
                     const EmptySurface& surface,
                     // I-O
                     RandomGPU& rng,
                     // Constants
                     const AlbedoMatData& matData,
                     const HitKey::Type& matId)
{
    return matData.dAlbedo[matId];
}

Vector3 ConstantEvaluate(// Input
                        const Vector3& wo,
                        const Vector3& wi,
                        const Vector3& pos,
                        const EmptySurface& surface,
                        // Constants
                        const AlbedoMatData& matData,
                        const HitKey::Type& matId)
{
    return matData.dAlbedo[matId];
}

Vector3 BarycentricShade(// Sampled Output
                         RayF& wo,
                         float& pdf,
                         // Input
                         const Vector3& wi,
                         const Vector3& pos,
                         const BarySurface& surface,
                         // I-O
                         RandomGPU& rng,
                         // Constants
                         const NullData& matData,
                         const HitKey::Type& matId)
{
    return surface.baryCoords;
}

Vector3 BarycentricEvaluate(// Input
                            const Vector3& wo,
                            const Vector3& wi,
                            const Vector3& pos,
                            const BarySurface& surface,
                            // Constants
                            const NullData& matData,
                            const HitKey::Type& matId)
{
    return matData.dAlbedo[matId];
}

Vector3 SphericalShade(// Sampled Output
                       RayF& wo,
                       float& pdf,
                       // Input
                       const Vector3& wi,
                       const Vector3& pos,
                       const SphrSurface& surface,
                       // I-O
                       RandomGPU& rng,
                       // Constants
                       const NullData& matData,
                       const HitKey::Type& matId)
{
    return Vector3f(cos(surface.sphrCoords[0]),
                    sin(surface.sphrCoords[1]),
                    0.0f);
}

Vector3 SphericalEvaluate(// Input
                          const Vector3& wo,
                          const Vector3& wi,
                          const Vector3& pos,
                          const SphrSurface& surface,
                          // Constants
                          const NullData& matData,
                          const HitKey::Type& matId)
{
    return Vector3f(cos(surface.sphrCoords[0]),
                    sin(surface.sphrCoords[1]),
                    0.0f);
}