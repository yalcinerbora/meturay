#pragma once
/**

Mandatory structures that are used
in tracing

*/

#include "Vector.h"

struct alignas(64) RayStack
{
	Vector3			position;
	float			currentMedium;
	Vector3			direction;

	
	Vector3			totalRadiance;
	

	unsigned int	pixelId;
	unsigned int	sampleId;	
};

struct HitRecord
{
	Vector3			bartyCoord;
	int				objectId;
	int				triangleId;
};

static_assert(sizeof(RayStack) == 64, "");