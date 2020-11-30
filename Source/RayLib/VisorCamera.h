#pragma once
/**
CPU Representation of the camera

No inheritance on CPU (Inheritance is on GPU)

There are not many different types of Camera
so it is somehow maintainable without using Interitance
*/

#include "Vector.h"
#include "HitStructs.h"

struct VisorCamera
{
    uint16_t    mediumIndex;
    HitKey      matKey;
    // World Space Lengths from camera
    Vector3     gazePoint;
    float       nearPlane;      // Distance from gaze
    Vector3     position;
    float       farPlane;       // Distance from gaze
    Vector3     up;
    float       apertureSize;
    Vector2     fov;            // degree
};