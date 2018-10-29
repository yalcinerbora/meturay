#pragma once
/**


*/

enum class PrimitiveDataType
{
	POSITION,
	NORMAL,
	UV,
	RADIUS,

	END
};

static constexpr const char* PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::END)] =
{
	"position",
	"normal",
	"uv",
	"radius"
};

static_assert(sizeof(PrimitiveDataTypeNames) / sizeof(const char*) ==
			  static_cast<int>(PrimitiveDataType::END), "String array and enum count mismatch.");

static constexpr const char* PrimitiveDataTypeToString(PrimitiveDataType t)
{
	return PrimitiveDataTypeNames[static_cast<int>(t)];
}


