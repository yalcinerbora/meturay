#pragma once
#pragma once
/**

Tracer error "Enumeration"

*/

#include <stdexcept>
#include "Error.h"

struct SceneError : public ErrorI
{
	public:
		enum Type
		{
			OK,
			// Common
			FILE_NOT_FOUND,
			ANIMATION_FILE_NOT_FOUND,
			// Not Found
			ACCELERATORS_ARRAY_NOT_FOUND,
			MATERIALS_ARRAY_NOT_FOUND,
			PRIMITIVES_ARRAY_NOT_FOUND,
			SURFACES_ARRAY_NOT_FOUND,
			SURFACE_DATA_ARRAY_NOT_FOUND,
			// No Logic
			NO_LOGIC_FOR_ACCELERATOR,
			NO_LOGIC_FOR_MATERIAL,
			NO_LOGIC_FOR_PRIMITIVE,
			// Id Errors
			DUPLICATE_ACCEL_ID,
			DUPLICATE_MATERIAL_ID,
			DUPLICATE_PRIMITIVE_ID,
			DUPLICATE_TRANSFORM_ID,
			DUPLICATE_SURFACE_DATA_ID,
			// 
			ACCEL_ID_NOT_FOUND,
			MATERIAL_ID_NOT_FOUND,
			PRIMITIVE_ID_NOT_FOUND,
			TRANSFORM_ID_NOT_FOUND,
			SURFACE_DATA_ID_NOT_FOUND,
			// Json parse errors
			LOGIC_MISMATCH,
			TYPE_MISMATCH,
			JSON_FILE_PARSE_ERROR,
			// Special Type Values
			UNKNOWN_TRANSFORM_TYPE,
			UNKNOWN_LIGHT_TYPE,

			// End
			END
		};

	private:
		Type			 type;

	public:
		// Constructors & Destructor 
					SceneError(Type);
					~SceneError() = default;

		operator	Type() const;
		operator	std::string() const override;
};

class SceneException : public std::runtime_error
{
	private:
		SceneError e;
	protected:
	public:
		SceneException(SceneError::Type t)		
			: std::runtime_error("")
			, e(t)
		{}
		operator SceneError() const { return e; };
};

inline SceneError::SceneError(SceneError::Type t)
	: type(t)
{}

inline SceneError::operator Type() const
{
	return type;
}

inline SceneError::operator std::string() const
{
	const char* const ErrorStrings[] =
	{
		"OK.",
		// Common
		"Scene file not found.",
		"Animation file not found.",		
		// Not Found
		"\"Accelerators\" array not found.",
		"\"Materials\" array not found.",
		"\"Primitives\" array not found.",
		"\"Surfaces\" array not found.",
		"\"SurfaceData\" array not found.",
		// No Logic
		"No logic found for that accelerator.",
		"No logic found for that material.",
		"No logic found for that primitive.",
		// Id Errors
		"Duplicate accelerator id.",
		"Duplicate material id.",
		"Duplicate primitive id.",
		"Duplicate transform id.",
		"Duplicate surface data id.",					
		// 
		"Accelerator id not found.",
		"Material id not found.",
		"Primitive id not found.",
		"Transform id not found.",
		"Surface data id not found.",		
		// Json Parse Errors
		"Logics does not match.",
		"JSON type does not match with required type.",
		"JSON file could not be parsed properly.",
		// Special Type Values
		"Transform type name is unknown.",
		"Light type name is unknown."
	};
	static_assert((sizeof(ErrorStrings) / sizeof(const char*)) == static_cast<size_t>(SceneError::END),
				  "Enum and enum string list size mismatch.");

	return ErrorStrings[static_cast<int>(type)];
}