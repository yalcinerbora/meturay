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
			MATERIALS_ARRAY_NOT_FOUND,
			PRIMITIVES_ARRAY_NOT_FOUND,
			SURFACES_ARRAY_NOT_FOUND,
			BASE_ACCELERATOR_NODE_NOT_FOUND,
			OUTSIDE_MAT_NODE_NOT_FOUND,
			// No Logic
			NO_LOGIC_FOR_ACCELERATOR,
			NO_LOGIC_FOR_MATERIAL,
			NO_LOGIC_FOR_PRIMITIVE,
			NO_LOGIC_FOR_SURFACE_DATA,
			// Id Errors
			DUPLICATE_MATERIAL_ID,
			DUPLICATE_PRIMITIVE_ID,
			DUPLICATE_TRANSFORM_ID,
			// Id not found
			MATERIAL_ID_NOT_FOUND,
			PRIMITIVE_ID_NOT_FOUND,
			TRANSFORM_ID_NOT_FOUND,
			// Json parse errors
			LOGIC_MISMATCH,
			TYPE_MISMATCH,
			JSON_FILE_PARSE_ERROR,
			// Special Type Values
			UNKNOWN_TRANSFORM_TYPE,
			UNKNOWN_LIGHT_TYPE,
			// Loading Surface Data
			SURFACE_DATA_TYPE_NOT_FOUND,
			SURFACE_DATA_INVALID_READ,
			// Some Mat/Accel Logic
			// may not support certain prims
			PRIM_ACCEL_MISMATCH,
			PRIM_MAT_MISMATCH,
			// Updating the Scene
			// Primitive Update Size Mismmatch
			PRIM_UPDATE_SIZE_MISMATCH,
			// Too many types than key system can handle
			TOO_MANY_ACCELERATOR_GROUPS,
			TOO_MANY_ACCELERATOR_IN_GROUP,
			TOO_MANY_MATERIAL_GROUPS,
			TOO_MANY_MATERIAL_IN_GROUP,
			// Misc
			TOO_MANY_SURFACE_ON_NODE,
			PRIM_MATERIAL_NOT_SAME_SIZE,
			PRIM_TYPE_NOT_CONSISTENT_ON_SURFACE,
			// Internal Errors
			INTERNAL_DUPLICATE_MAT_ID,
			INTERNAL_DUPLICATE_ACCEL_ID,
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
		"\"Materials\" array not found.",
		"\"Primitives\" array not found.",
		"\"Surfaces\" array not found.",
		"\"BaseAccelerator\" node not found.",
		"\"OutsideMaterial\" node not found.",
		// No Logic
		"No logic found for that accelerator.",
		"No logic found for that material.",
		"No logic found for that primitive.",
		"No logic found for loading that surface data.",
		// Id Errors
		"Duplicate material id.",
		"Duplicate primitive id.",
		"Duplicate transform id.",
		//
		"Material id not found.",
		"Primitive id not found.",
		"Transform id not found.",
		// Json Parse Errors
		"Logics does not match.",
		"JSON type does not match with required type.",
		"JSON file could not be parsed properly.",
		// Special Type Values
		"Transform type name is unknown.",
		"Light type name is unknown.",
		// Loading Surface Data
		"Surface data type not found.",
		"Surface data unknown type.",
		// Some Mat/Accel Logic
		// may not support certain prims
		"Primitive-Material mismatch.",
		"Primitive-Accelerator mismatch.",
		// Updating the scene
		// Primitive Update Size Mismmatch
		"Updating primitive has more nodes than older itself.",
		// Too many types than key system can handle
		"Accelerator groups required for this scene exceeds limit.",
		"Accelerator groups required for this scene exceeds limit.",
		"Accelerator groups required for this scene exceeds limit.",
		"Accelerator groups required for this scene exceeds limit.",
		// Misc
		"Too many data/material pairs per surface node.",
		"Prim/Material pairs on surface node does not have same size.",
		"Primitive types are not consistent in a surface.",
		// Internal Errors
		"Internal Error, Duplicate material id",
		"Internal Error, Duplicate accelerator id"
	};
	static_assert((sizeof(ErrorStrings) / sizeof(const char*)) == static_cast<size_t>(SceneError::END),
				  "Enum and enum string list size mismatch.");

	return ErrorStrings[static_cast<int>(type)];
}