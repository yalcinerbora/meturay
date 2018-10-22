#pragma once

#include <vector>

#include "Vector.h"
#include "Matrix.h"
#include "SceneError.h"
#include "Types.h"

enum class LightType
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

using TransformStruct = Matrix4x4;

struct SurfaceStruct
{
	uint32_t transformId;
	uint32_t materialId;
	uint32_t primitiveId;
	uint32_t acceleratorId;
	uint32_t dataId;

	static bool SortComparePrimitive(const SurfaceStruct* a,
									 const SurfaceStruct* b);
	static bool SortComparePrimAccel(const SurfaceStruct* a,
									 const SurfaceStruct* b);
	static bool SortComparePrimMaterial(const SurfaceStruct* a,
										const SurfaceStruct* b);
};

struct PrimitiveStruct
{
	struct PrimitiveData
	{
		std::string logic;		// Logic of the data (used by accelerator / material)
		uint32_t intake;		// Intake index (if data is stored multiple linear portions)
		uint32_t stride;		// Byte stride of the data
		uint32_t offset;		// Byte offset from the start of the index
		DataType type;			// Data type
	};

	// Members
	std::vector<PrimitiveData>	dataDefinitions;
	std::string					type;
	uint32_t					id;
};

inline bool SurfaceStruct::SortComparePrimitive(const SurfaceStruct* a,
												const SurfaceStruct* b)
{
	return a->primitiveId < b->primitiveId;
}

inline bool SurfaceStruct::SortComparePrimAccel(const SurfaceStruct* a,
												const SurfaceStruct* b)
{
	return (a->primitiveId < b->primitiveId ||
		   (a->primitiveId == b->primitiveId && a->acceleratorId < b->acceleratorId));
}

inline bool SurfaceStruct::SortComparePrimMaterial(const SurfaceStruct* a,
												   const SurfaceStruct* b)
{
	return (a->primitiveId < b->primitiveId ||
		   (a->primitiveId == b->primitiveId && a->materialId < b->materialId));
}