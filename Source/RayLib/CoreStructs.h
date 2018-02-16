#pragma once
/**

Mandatory structures that are used
in tracing

*/

#include "Vector.h"

struct alignas(64) RayStack
{
	Vector3			position;
	Vector3			direction;
	Vector3			totalRadiance;
	float			currentMedium;

	unsigned int	pixelId;
	unsigned int	sampleId;
	
	Vector3			bartyCoord;
	int				objectId;
	//int				triangleId;
};

struct HitRecord
{

};

static_assert(sizeof(RayStack) == 64, "");