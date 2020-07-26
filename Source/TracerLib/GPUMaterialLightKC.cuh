#pragma once

#include <cuda.h>

#include "RayLib/Vector.h"
#include "RayLib/HitStructs.h"
#include "SurfaceStructs.h"
#include "TextureReference.cuh"

class GPUMedium;
struct TexCoords;

struct LightMatData
{
    Vector3* dRadiances;
};

struct LightMatTexData
{
    TexRef<2, Vector3>* dRadianceTextures;
};

struct LightMatCubeData
{
    TexCubeRef<Vector3>* dCubeTextures;
};

template <class Surface>
__device__ inline
Vector3 EmitLight(// Input
                  const Vector3& wo,
                  const Vector3& pos,
                  const GPUMedium& m,
                  //
                  const Surface& surface,
                  const TexCoords* uvs,
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
                     const GPUMedium& m,
                     //
                     const BasicUVSurface& surface,
                     const TexCoords* uvs,
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
                      const GPUMedium& m,
                      //
                      const EmptySurface& surface,
                      const TexCoords* uvs,
                      // Constants
                      const LightMatCubeData& matData,
                      const HitKey::Type& matId)
{
    // Gen Directional vector
    return matData.dCubeTextures[matId](pos);   
}