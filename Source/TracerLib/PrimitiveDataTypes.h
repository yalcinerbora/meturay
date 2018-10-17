#pragma once


enum class PrimitiveBasicDataType
{
	POSITION,
	NORMAL,
	UV,

	END
};

static constexpr const char* PrimitiveBasicDataTypeNames[static_cast<int>(PrimitiveBasicDataType::END)] =
{
	"position",
	"normal",
	"uv"
};

static_assert(sizeof(PrimitiveBasicDataTypeNames) / sizeof(const char*) ==
			  static_cast<int>(PrimitiveBasicDataType::END), "String array and enum count mismatch.");

static constexpr const char* PrimBasicDataTypeToString(PrimitiveBasicDataType t)
{
	return PrimitiveBasicDataTypeNames[static_cast<int>(t)];
}

