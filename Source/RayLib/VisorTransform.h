#pragma once
/**
CPU Representation of the camera

No inheritance on CPU (Inheritance is on GPU)

There are not many different types of Camera
so it is somehow maintainable without using Inheritance
*/

#include "Vector.h"
#include "HitStructs.h"

#include <sstream>

struct VisorTransform
{
    Vector3f position;
    Vector3f gazePoint;
    Vector3f up;
};

//struct VisorCamera
//{
//    uint16_t    mediumIndex;
//    HitKey      matKey;
//    // World Space Lengths from camera
//    Vector3     gazePoint;
//    float       nearPlane;      // Distance from gaze
//    Vector3     position;
//    float       farPlane;       // Distance from gaze
//    Vector3     up;
//    float       apertureSize;
//    Vector2     fov;            // degree
//};

inline std::string VisorTransformToString(const VisorTransform& c)
{
    std::stringstream s;
    //s << "M Index  : " << c.mediumIndex << std::endl;
    //s << "Key      : " << std::hex << c.matKey.value << std::dec << std::endl;
    s << "Gaze     : [" << c.gazePoint[0] << ", "
                        << c.gazePoint[1]  << ", "
                        << c.gazePoint[2] << "]" << std:: endl;
    s << "Pos      : [" << c.position[0] << ", "
                        << c.position[1]  << ", "
                        << c.position[2] << "]" << std:: endl;
    s << "Up       : [" << c.up[0] << ", "
                        << c.up[1]  << ", "
                        << c.up[2] << "]" << std:: endl;
    //s << "Near Far : [" << c.nearPlane << ", "
    //                    << c.farPlane << "]" << std::endl;
    //s << "Fov      : [" << c.fov[0] << ", "
    //                    << c.fov[1] << "]" << std::endl;
    //s << "Aperture : " << c.apertureSize << std::endl;
    return s.str();
}