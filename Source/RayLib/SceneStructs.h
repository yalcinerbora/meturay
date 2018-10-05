#pragma once

#include "Vector.h"
#include "Matrix.h"

#include "SceneError.h"

enum LightType
{
	POINT,
	DIRECTIONAL,
	SPOT,
	RECTANGULAR
};

struct LightStruct
{
	LightType t;
	union
	{
		struct
		{
			Vector3f	position;
			float		intensity;
			Vector3f	color;
		} point;
		struct
		{
			Vector3f	direction;
			float		intensity;
			Vector3f	color;
		} directional;
		struct
		{
			Vector3f	position;
			float		coverageAngle;
			Vector3f	direction;
			float		falloffAngle;
			Vector3f	color;
			float		intensity;
		} spot;
		struct
		{
			Vector3f	position;
			float		red;
			Vector3f	edge0;
			float		green;
			Vector3f	edge1;
			float		blue;
			float		intensity;
		} rectangular;
	};
};

typedef Matrix4x4 TransformStruct;

//
struct AcceleraorStruct
{
	uint32_t id;
	uint32_t type;
};

struct SurfaceStruct
{
	uint32_t transformId;
	uint32_t materialId;
	uint32_t primitiveId;
	uint32_t acceleratorId;
	uint32_t dataId;	
};