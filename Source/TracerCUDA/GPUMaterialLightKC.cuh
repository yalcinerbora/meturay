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

struct LightMatCubeData
{
    const TexCubeRef<Vector3>* dCubeTextures;
};

__device__ inline
Vector3 EmitLight(// Input
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
Vector3 EmitLightTex(// Input
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
Vector3 EmitLightCube(// Input
                      const Vector3& wo,
                      const Vector3& pos,
                      const GPUMediumI& m,
                      //
                      const EmptySurface& surface,
                      // Constants
                      const LightMatCubeData& matData,
                      const HitKey::Type& matId)
{
    // Gen Directional vector
    return matData.dCubeTextures[matId](pos);
}

__device__ inline
Vector3 EmitLightSkySphere(// Input
                           const Vector3& wo,
                           const Vector3& pos,
                           const GPUMediumI& m,
                           //
                           const EmptySurface& surface,
                           // Constants
                           const LightMatTexData& matData,
                           const HitKey::Type& matId)
{
    // Convert to Spherical Coordinates
    Vector2f tehtaPhi = Utility::CartesianToSphericalUnit(wo);

    // Normalize to generate UV [0, 1]
    // tetha range [-pi, pi]
    float u = (tehtaPhi[0] + MathConstants::Pi) * 0.5f / MathConstants::Pi;
    // phi range [0, pi]
    float v = tehtaPhi[1] / MathConstants::Pi;

    // Gen Directional vector
    return matData.dRadianceTextures[matId](Vector2(u,v));
}