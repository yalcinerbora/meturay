#pragma once

#include "Vector.h"

struct CameraPerspective
{
	// World Space Lengths from camera
	Vector3		gazePoint;
	float		near;	
	Vector3		position;
	float		far;
	Vector3		up;
	float		apertureSize;
	Vector2		fov;
};