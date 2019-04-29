#pragma once

#include "Vector.h"

struct CameraPerspective
{
    // World Space Lengths from camera
    Vector3     gazePoint;
    float       nearPlane;
    Vector3     position;
    float       farPlane;
    Vector3     up;
    float       apertureSize;
    Vector2     fov;
};