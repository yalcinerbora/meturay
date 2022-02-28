#pragma once

#include "RayLib/Vector.h"
#include "RayLib/Constants.h"
#include "RayLib/CoordinateConversion.h"

namespace GPUDataStructCommon
{
    __device__
    Vector2f DirToDiscreteCoords(const Vector3f& worldDir);

    __device__
    Vector2f DirToDiscreteCoords(float& pdf, const Vector3f& worldDir);

    __device__
    Vector3f DiscreteCoordsToDir(float& pdf, const Vector2f& discreteCoords);

    __device__
    Vector3f DiscreteCoordsToDir(const Vector2f& discreteCoords);
}

__device__ __forceinline__
Vector2f GPUDataStructCommon::DirToDiscreteCoords(const Vector3f& worldDir)
{
    float pdf = 0.0f;
    return DirToDiscreteCoords(pdf, worldDir);
}

__device__ __forceinline__
Vector2f GPUDataStructCommon::DirToDiscreteCoords(float& pdf, const Vector3f& worldDir)
{
    Vector3 wZup = Vector3(worldDir[2], worldDir[0], worldDir[1]);
    // Convert to Spherical Coordinates
    Vector2f thetaPhi = Utility::CartesianToSphericalUnit(wZup);
    // Normalize to generate UV [0, 1]
    // theta range [-pi, pi]
    float u = (thetaPhi[0] + MathConstants::Pi) * 0.5f / MathConstants::Pi;
    // If we are at edge point (u == 1) make it zero since
    // piecewise constant function will not have that pdf (out of bounds)
    u = (u == 1.0f) ? 0.0f : u;
    // phi range [0, pi]
    float v = 1.0f - (thetaPhi[1] / MathConstants::Pi);
    // If (v == 1) then again pdf of would be out of bounds.
    // make it inbound
    v = (v == 1.0f) ? (v - MathConstants::SmallEpsilon) : v;

    // Pre-Convert to solid angle pdf
    // http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources.html
    float sinPhi = sin(thetaPhi[1]);
    if(sinPhi == 0.0f) pdf = 0.0f;
    else pdf = pdf / (2.0f * MathConstants::Pi * MathConstants::Pi * sinPhi);
    return Vector2f(u,v);
}

__device__ __forceinline__
Vector3f GPUDataStructCommon::DiscreteCoordsToDir(const Vector2f& discreteCoords)
{
    float pdf = 0.0f;
    return DiscreteCoordsToDir(pdf, discreteCoords);
}

__device__ __forceinline__
Vector3f GPUDataStructCommon::DiscreteCoordsToDir(float& pdf, const Vector2f& discreteCoords)
{
    // Convert the Local 2D cartesian coords to spherical coords
    const Vector2f& uv = discreteCoords;
    Vector2f thetaPhi = Vector2f(// [-pi, pi]
                                 (uv[0] * MathConstants::Pi * 2.0f) - MathConstants::Pi,
                                 // [0, pi]
                                 (1.0f - uv[1]) * MathConstants::Pi);
    Vector3 dirZUp = Utility::SphericalToCartesianUnit(thetaPhi);
    // Spherical Coords calculates as Z up change it to Y up
    Vector3 dirYUp = Vector3(dirZUp[1], dirZUp[2], dirZUp[0]);
    // Convert to solid angle pdf
    // http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources.html
    float sinPhi = sin(thetaPhi[1]);
    if(sinPhi == 0.0f) pdf = 0.0f;
    else pdf = pdf / (2.0f * MathConstants::Pi * MathConstants::Pi * sinPhi);
    return dirYUp;
}