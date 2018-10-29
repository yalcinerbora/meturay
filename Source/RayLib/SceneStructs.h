#pragma once

#include <vector>
#include <array>
#include <set>

#include "Vector.h"
#include "Matrix.h"
#include "SceneError.h"
#include "Types.h"

using SurfaceDataId = uint32_t;
using MaterialId = uint32_t;
using IdPairings = std::array<std::pair<MaterialId, SurfaceDataId>, SceneConstants::MaxSurfacePerAccelerator>;

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
	uint32_t		transformId;	
	uint32_t		primitiveId;
	uint32_t		acceleratorId;
	IdPairings		matDataPairs;
	int8_t			pairCount;

	//bool			operator<(const SurfaceStruct& right) const;
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

//inline bool SurfaceStruct::operator<(const SurfaceStruct& right)
//{
//	bool case0 = primitiveId < right.primitiveId;
//	bool case1 = (primitiveId == right.primitiveId &&
//				  acceleratorId < right.acceleratorId);
//	return (case0 || case1);
//}
