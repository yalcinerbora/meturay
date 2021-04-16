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
    return matData.dRadianceTextures[matId](surface.uv);
}

__device__ inline
Vector3 EmitSkySphere(// Input
                      const Vector3& wo,
                      const Vector3& pos,
                      const GPUMediumI& m,
                      //
                      const BasicSurface& surface,
                      // Constants
                      const LightMatTexData& matData,
                      const HitKey::Type& matId)
{
    //printf("WtT: %f, %f, %f, %f\n",
    //       surface.worldToTangent[0],
    //       surface.worldToTangent[1],
    //       surface.worldToTangent[2],
    //       surface.worldToTangent[3]);

    // Convert Y up from Z up
    // Also invert since that direction is used to sample HDR texture
    Vector3 woTrans = GPUSurface::ToTangent(wo, surface.worldToTangent);
    Vector3 woZup = -Vector3(woTrans[2], woTrans[0], woTrans[1]);

    // Convert to Spherical Coordinates
    Vector2f tehtaPhi = Utility::CartesianToSphericalUnit(woZup);

    // Normalize to generate UV [0, 1]
    // tetha range [-pi, pi]
    float u = (tehtaPhi[0] + MathConstants::Pi) * 0.5f / MathConstants::Pi;
    // phi range [0, pi]
    float v = 1.0f - (tehtaPhi[1] / MathConstants::Pi);

    //printf("Received Light from (%f, %f, %f)\n"
    //       "Zup     : %f, %f, %f\n"
    //       "UV      : %f, %f\n",
    //       wo[0], wo[1], wo[2],
    //       woZup[0], woZup[1], woZup[2],
    //       u, v);

    // Gen Directional vector
    return matData.dRadianceTextures[matId](Vector2(u,v));
}