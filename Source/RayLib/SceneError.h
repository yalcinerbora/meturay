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
			// No Logic
			NO_LOGIC_FOR_ACCELERATOR,
			NO_LOGIC_FOR_MATERIAL,
			NO_LOGIC_FOR_PRIMITIVE,
			// Logics does not combine
			LOGIC_MISMATCH,
			// Json Errors
			TYPE_MISMATCH,
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
		// No Logic
		"No logic found for that accelerator.",
		"No logic found for that material.",
		"No logic found for that primitive.",
		// Logics does not combine
		"Logics does not match.",
		"JSON type does not match with required type.",
		// Special Type Values
		"Transform type name is unknown.",
		"Light type name is unknown."
	};
	static_assert((sizeof(ErrorStrings) / sizeof(const char*)) == static_cast<size_t>(SceneError::END),
				  "Enum and enum string list size mismatch.");

	return ErrorStrings[static_cast<int>(type)];
}