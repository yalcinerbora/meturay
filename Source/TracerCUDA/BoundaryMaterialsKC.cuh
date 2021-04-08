#pragma once

#include <cuda.h>

#include "RayLib/Vector.h"
#include "RayLib/HitStructs.h"
#include "RayLib/CoordinateConversion.h"

#include "GPUSurface.h"
#include "TextureReference.cuh"

class GPUMediumI;
struct TexCoords;

struct LightMatData
{
    const Vector3* dRadiances;
};

struct LightMatTexData
{
    const TextureRef<2, Vector3>* dRadianceTextures;
};

__device__ inline
Vector3 EmitConstant(// Input
                     const Vector3& wo,
                     const Vector3& pos,
                     const GPUMediumI& m,
                     //
                     const EmptySurface& surface,
                     // Constants
                     const LightMatData& matData,
                     const HitKey::Type& matId)
{
    return matData.dRadiances[matId];
}

__device__ inline
Vector3 EmitTextured(// Input
                     const Vector3& wo,
                     const Vector3& pos,
                     const GPUMediumI& m,
                     //
                     const UVSurface& surface,
                     // Constants
                     const LightMatTexData& matData,
                     const HitKey::Type& matId)
{
    // If cubemap
    return matData.dRadianceTextures[matId](surface.uv);
}

__device__ inline
Vector3 EmitSkySphere(// Input
                      const Vector3& wo,
                      const Vector3& pos,
                      const GPUMediumI& m,
                      //
                      const EmptySurface& surface,
                      // Constants
                      const LightMatTexData& matData,
                      const HitKey::Type& matId)
{
    // Convert Y up from Z up
    Vector3 woZup = Vector3(wo[2], wo[0], wo[1]);

    // Convert to Spherical Coordinates
    Vector2f tehtaPhi = Utility::CartesianToSphericalUnit(woZup);

    // Normalize to generate UV [0, 1]
    // tetha range [-pi, pi]
    float u = (tehtaPhi[0] + MathConstants::Pi) * 0.5f / MathConstants::Pi;
    // phi range [0, pi]
    float v = tehtaPhi[1] / MathConstants::Pi;

    // Gen Directional vector
    return matData.dRadianceTextures[matId](Vector2(u,v));
}